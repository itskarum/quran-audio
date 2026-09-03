"""End-to-end restoration pipeline.

Stage order per channel (each stage preserves sample count and alignment):

  1. DC removal, analysis (noise floor, pauses, hum, bandwidth, clipping)
  2. hum notches            (only harmonics that were actually measured)
  3. declip                 (auto: only when clipping was detected)
  4. declick / decrackle    (AR-model detection, LSAR repair)
  5. broadband denoise      (classical OM-LSA or DeepFilterNet3)
  6. voice EQ               (rumble high-pass, bandwidth low-pass, tonal balance)
  7. leveler
then, jointly over channels:
  8. loudness normalisation + true-peak limiting

The "residual" is (input after DC removal) minus (output of stage 5): it is
exactly the material the tool decided was noise. Listening to it is the
fastest way to confirm that no voice was removed.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .analysis import Analysis, analyze, speech_spectrum
from .audio_io import fit_length, load, remove_dc, resample, resampled_length, save, to_mono
from .declick import declick, declip
from .denoise import DenoiseSettings, denoise, denoise_linked
from .dynamics import integrated_loudness, limiter, leveler_gain, normalize_loudness, true_peak_dbtp
from .eq import apply_fir, design_tonal_balance, highpass, lowpass, rumble_cutoff
from .hum import remove_hum

Log = Callable[[str], None]


@dataclass
class Settings:
    preset: str = "standard"
    channels: str = "mono"               # mono | keep
    mono_strategy: str = "auto"          # auto | best | coherent-sum | mix | left | right
    output_sr: int | None = None         # None = same as input
    output_subtype: str | None = None    # libsndfile subtype, e.g. PCM_24
    dc: bool = True
    hum: bool = True
    hum_min_prominence_db: float = 6.0
    declick: bool = True
    declick_threshold: float = 6.0
    declick_max_ms: float = 2.0
    decrackle: bool = False
    decrackle_threshold: float = 4.5
    decrackle_max_ms: float = 0.6
    declip: str = "auto"                 # auto | on | off
    declip_min_runs: int = 20
    denoise: str = "auto"                # auto (= classical) | classical | dfn | off
    denoise_floor_db: float = -18.0      # deepest floor, used at <= 16 dB measured SNR
    denoise_floor_min_db: float = -12.0  # lightest floor, used at 28 dB measured SNR (half of it at 40 dB)
    denoise_adaptive: bool = True        # scale the floor with the measured SNR
    denoise_hf_floor_db: float = -40.0
    denoise_fusion: str = "soft"
    denoise_absence_prior: float = 0.5
    dfn_atten_lim_db: float | None = 30.0
    dfn_model: str | None = None
    highpass: bool = True
    lowpass: bool = True
    tonal_balance: bool = True
    tonal_strength: float = 0.5
    tonal_max_db: float = 6.0
    tonal_reference: str = "recitation"   # recitation | speech | path to a clean reference recording
    tail_preserve: bool = True
    leveler: bool = False
    leveler_range_db: float = 3.0
    leveler_attack_s: float = 0.6
    leveler_release_s: float = 4.0
    leveler_window_s: float = 0.4
    target_lufs: float | None = -18.0    # None = no loudness normalisation
    true_peak_db: float = -1.0
    residual: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, dict] = {
    "gentle": dict(declick_threshold=7.0, decrackle=False, denoise_floor_db=-12.0, denoise_floor_min_db=-8.0,
                   denoise_hf_floor_db=-30.0, dfn_atten_lim_db=20.0, tonal_strength=0.3,
                   tonal_max_db=4.0),
    "standard": {},
    "strong": dict(declick_threshold=5.0, decrackle=True, denoise_floor_db=-28.0, denoise_floor_min_db=-16.0,
                   denoise_hf_floor_db=-50.0, dfn_atten_lim_db=None, tonal_strength=0.7,
                   tonal_max_db=9.0, leveler=True, leveler_range_db=4.0),
    # broadcast: for playback on small speakers and streaming, where an even
    # level matters more than the reciter's dynamics. This is compression.
    "broadcast": dict(leveler=True, leveler_range_db=6.0, leveler_attack_s=0.3, leveler_release_s=2.0,
                      leveler_window_s=0.2, target_lufs=-16.0),
}


def adaptive_floor_db(preset_floor_db: float, floor_min_db: float, snr_db: float) -> float:
    """Full preset strength at <= 16 dB measured SNR, the lightest floor at
    >= 28 dB, linear in between. A masking denoiser attenuates every
    harmonic that sits near the noise; on a clean transfer that costs more
    voice than it removes noise, so the floor rises with the SNR."""
    if snr_db <= 28.0:
        w = float(np.clip((28.0 - snr_db) / 12.0, 0.0, 1.0))
        return float(floor_min_db + w * (preset_floor_db - floor_min_db))
    # cleaner still: keep easing off up to 40 dB, where half the lightest floor remains
    w = float(np.clip((snr_db - 28.0) / 12.0, 0.0, 1.0))
    return float(floor_min_db * (1.0 - 0.5 * w))


def make_settings(preset: str = "standard", overrides: dict | None = None) -> Settings:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    values = {**PRESETS[preset], "preset": preset, **(overrides or {})}
    return Settings(**values)


@dataclass
class Result:
    samples: np.ndarray                  # (n, channels) at `sample_rate`
    sample_rate: int
    residual: np.ndarray | None          # (n, channels) at `sample_rate`, or None
    report: dict = field(default_factory=dict)


class _Timer:
    def __init__(self, log: Log | None) -> None:
        self.log = log
        self.timings: dict[str, float] = {}

    def run(self, name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        self.timings[name] = round(self.timings.get(name, 0.0) + dt, 3)
        if self.log:
            self.log(f"{name}: {dt:.1f}s")
        return out


def _pick_backend(settings: Settings, log: Log | None):
    """Return (backend_name, dfn_denoiser_or_None, note).

    `auto` means the classical backend: DeepFilterNet also dereverberates,
    and on reverberant recitation it removes most of the room along with
    the noise, which is a change of character the operator must opt into."""
    if settings.denoise == "off":
        return "off", None, ""
    if settings.denoise in ("classical", "auto"):
        return "classical", None, ""
    if settings.denoise == "dfn":
        from .denoise_dfn import DFNDenoiser
        return "dfn", DFNDenoiser(settings.dfn_model), ""
    raise ValueError(f"unknown denoise backend {settings.denoise!r}")


def _process_group(chans: list[np.ndarray], sr: int, settings: Settings, backend: str, dfn,
                   timer: _Timer) -> tuple[list[np.ndarray], list[np.ndarray], list[dict], Analysis]:
    """Run stages 1-7 on one channel, or on a linked pair. Every decision
    (analysis, denoiser gains, EQ, leveler gain) is taken on the mid of the
    group and applied to all its channels, so a stereo image never wobbles.
    Sample-domain repairs (hum, clicks, clipping) are per channel because
    the disturbance itself differs per channel. Returns (pre_eq, output,
    reports, analysis)."""
    n = len(chans[0])
    reps: list[dict] = [{} for _ in chans]
    xs: list[np.ndarray] = []
    for c, x in enumerate(chans):
        if settings.dc:
            x, offset = remove_dc(x)
            reps[c]["dc_offset_removed"] = round(offset, 6)
        xs.append(x)

    def mid_of(sig: list[np.ndarray]) -> np.ndarray:
        return sig[0] if len(sig) == 1 else np.mean(np.stack(sig, axis=0), axis=0)

    an = timer.run("analysis", lambda: analyze(mid_of(xs), sr, hum_search=settings.hum))
    for c in range(len(xs)):
        reps[c]["notes"] = list(an.notes)
        if settings.hum and an.hum.detected:
            xs[c], notches = timer.run("hum", lambda x=xs[c]: remove_hum(
                x, sr, an.hum, settings.hum_min_prominence_db, pause_ranges=an.pause_ranges))
            reps[c]["hum_notches"] = notches
        else:
            reps[c]["hum_notches"] = []
        do_declip = settings.declip == "on" or (settings.declip == "auto" and an.clipped_runs >= settings.declip_min_runs)
        if do_declip:
            xs[c], r = timer.run("declip", lambda x=xs[c]: declip(x, sr))
            reps[c]["declip"] = r.to_dict()
        if settings.declick:
            xs[c], r = timer.run("declick", lambda x=xs[c]: declick(x, sr, settings.declick_threshold, settings.declick_max_ms))
            reps[c]["declick"] = r.to_dict()
        if settings.decrackle:
            xs[c], r = timer.run("decrackle", lambda x=xs[c]: declick(x, sr, settings.decrackle_threshold, settings.decrackle_max_ms))
            reps[c]["decrackle"] = r.to_dict()

    # noise profile / bandwidth after the sample-domain repairs
    ref = mid_of(xs)
    an2 = timer.run("analysis", lambda: analyze(ref, sr, hum_search=False))

    if backend == "classical" and not an2.noise_measurable:
        info = {"backend": "skipped", "reason": "no measurable stationary noise", "notes": list(an2.notes)}
    elif backend == "classical":
        floor_db = settings.denoise_floor_db
        if settings.denoise_adaptive:
            floor_db = adaptive_floor_db(settings.denoise_floor_db, settings.denoise_floor_min_db, an2.snr_db)
        ds = DenoiseSettings(floor_db=floor_db, hf_floor_db=settings.denoise_hf_floor_db,
                             bandwidth_hz=an2.bandwidth_hz, fusion=settings.denoise_fusion,
                             speech_absence_prior=settings.denoise_absence_prior,
                             tail_preserve=settings.tail_preserve)
        if len(xs) == 1:
            y, info = timer.run("denoise", lambda: denoise(xs[0], sr, an2, ds))
            xs = [y]
        else:
            ys, info = timer.run("denoise", lambda: denoise_linked(ref, xs, sr, an2, ds))
            xs = list(ys)
        info["snr_db_measured"] = round(float(an2.snr_db), 2)
        info["floor_db_effective"] = round(float(floor_db), 2)
    elif backend == "dfn":
        from .denoise_dfn import denoise_dfn
        results = [timer.run("denoise", lambda x=x: denoise_dfn(x, sr, settings.dfn_atten_lim_db, denoiser=dfn)) for x in xs]
        xs = [r[0] for r in results]
        info = dict(results[0][1])
        if len(xs) > 1:
            info["notes"] = info.get("notes", []) + ["DeepFilterNet processes channels independently (no gain linking)"]
    else:
        info = {"backend": "off"}
    for c in range(len(xs)):
        reps[c]["denoise"] = info
    pres = list(xs)

    eq_rep: dict = {}
    if settings.highpass:
        fc = rumble_cutoff(an2.low_edge_hz)
        xs = [timer.run("eq", lambda x=x: highpass(x, sr, fc)) for x in xs]
        eq_rep["highpass_hz"] = round(fc, 1)
    if settings.lowpass and an2.bandwidth_hz < 0.4 * sr:
        fc = min(an2.bandwidth_hz * 1.15, 0.45 * sr)
        xs = [timer.run("eq", lambda x=x: lowpass(x, sr, fc)) for x in xs]
        eq_rep["lowpass_hz"] = round(fc, 1)
    if settings.tonal_balance:
        # measured on the denoised voice: what it sounds like now, with
        # nothing assumed below the residual noise
        fq, sp, nz = timer.run("analysis", lambda: speech_spectrum(mid_of(pres), sr))
        clean_psd = np.maximum(sp - nz, nz)
        taps, bands = design_tonal_balance(clean_psd, fq, sr, an2.low_edge_hz, an2.bandwidth_hz,
                                           settings.tonal_strength, settings.tonal_max_db,
                                           reference=settings.tonal_reference)
        if taps is not None:
            xs = [timer.run("eq", lambda x=x: apply_fir(x, taps)) for x in xs]
        eq_rep["tonal_balance"] = bands
        eq_rep["tonal_reference"] = settings.tonal_reference
    for c in range(len(xs)):
        reps[c]["eq"] = eq_rep

    if settings.leveler:
        gain, info = timer.run("leveler", lambda: leveler_gain(mid_of(xs), sr, settings.leveler_range_db,
                                                               attack_s=settings.leveler_attack_s,
                                                               release_s=settings.leveler_release_s,
                                                               window_s=settings.leveler_window_s))
        if gain is not None:
            xs = [x * gain for x in xs]
        for c in range(len(xs)):
            reps[c]["leveler"] = info

    if any(len(x) != n for x in xs) or any(len(x) != n for x in pres):
        raise AssertionError("internal error: a stage changed the sample count")
    return pres, xs, reps, an


def enhance_signal(samples: np.ndarray, sr: int, settings: Settings | None = None,
                   log: Log | None = None) -> Result:
    """Restore `samples` (1-D, or (n, channels)) at sample rate `sr`."""
    settings = settings or make_settings()
    timer = _Timer(log)
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_in, ch_in = x.shape
    stereo_rep = None
    if settings.channels == "mono":
        mono, strategy, stereo_rep = to_mono(x, settings.mono_strategy, sr)
        groups = [[mono]]
    elif ch_in == 2:
        strategy = "keep-linked"
        groups = [[x[:, 0], x[:, 1]]]
    else:
        strategy = "keep"
        groups = [[x[:, c]] for c in range(ch_in)]
    if log and stereo_rep is not None:
        log(f"stereo: {stereo_rep.strategy} ({stereo_rep.reason})")

    backend, dfn, note = _pick_backend(settings, log)
    outs, pres, reps, analyses, chans = [], [], [], [], []
    for g, group in enumerate(groups):
        if log and len(groups) > 1:
            log(f"channel group {g + 1}/{len(groups)}")
        pre, out, rep, an = _process_group(group, sr, settings, backend, dfn, timer)
        pres.extend(pre)
        outs.extend(out)
        reps.extend(rep)
        analyses.append(an)
        chans.extend(group)
    y = np.stack(outs, axis=1)
    inputs = np.stack([remove_dc(c)[0] if settings.dc else c for c in chans], axis=1)
    residual = inputs - np.stack(pres, axis=1) if settings.residual else None

    if settings.target_lufs is not None:
        y, loud = timer.run("loudness", lambda: normalize_loudness(y, sr, settings.target_lufs, settings.true_peak_db))
    else:
        y, lim = timer.run("loudness", lambda: limiter(y, sr, settings.true_peak_db))
        loud = {"loudness_after_lufs": integrated_loudness(y, sr), "limiter": lim,
                "true_peak_after_dbtp": round(true_peak_dbtp(y), 2)}

    out_sr = settings.output_sr or sr
    if out_sr != sr:
        n_out = resampled_length(n_in, sr, out_sr)
        y = np.stack([fit_length(resample(y[:, c], sr, out_sr), n_out) for c in range(y.shape[1])], axis=1)
        if residual is not None:
            residual = np.stack([fit_length(resample(residual[:, c], sr, out_sr), n_out) for c in range(residual.shape[1])], axis=1)
    after = timer.run("analysis", lambda: analyze(y[:, 0], out_sr, hum_search=False))

    report = {
        "tool": "quran-audio", "version": __version__,
        "settings": settings.to_dict(),
        "input": {"sample_rate": sr, "channels": ch_in, "samples": n_in, "duration_s": round(n_in / sr, 3),
                  "mono_strategy": strategy},
        "stereo": stereo_rep.to_dict() if stereo_rep is not None else None,
        "denoise_backend": backend, "notes": [note] if note else [],
        "analysis_before": analyses[0].to_dict(),
        "channels": reps,
        "loudness": loud,
        "analysis_after": after.to_dict(),
        "output": {"sample_rate": out_sr, "channels": int(y.shape[1]), "samples": int(y.shape[0]),
                   "duration_s": round(y.shape[0] / out_sr, 3), "peak_dbfs": round(float(20 * np.log10(np.max(np.abs(y)) + 1e-12)), 2)},
        "timing_s": timer.timings,
        "length_preserved": int(y.shape[0]) == resampled_length(n_in, sr, out_sr),
    }
    return Result(y, out_sr, residual, report)


def enhance_file(input_path: str | Path, output_path: str | Path, settings: Settings | None = None,
                 residual_path: str | Path | None = None, log: Log | None = None) -> dict:
    settings = settings or make_settings()
    if residual_path is not None:
        settings = Settings(**{**settings.to_dict(), "residual": True})
    t0 = time.perf_counter()
    audio = load(input_path)
    if log:
        log(f"loaded {input_path}: {audio.sample_rate} Hz, {audio.n_channels} ch, {audio.duration:.1f} s, {audio.subtype}")
    result = enhance_signal(audio.samples, audio.sample_rate, settings, log)
    written = save(output_path, result.samples, result.sample_rate, settings.output_subtype)
    report = result.report
    report["input"]["path"] = str(input_path)
    report["input"]["subtype"] = audio.subtype
    report["output"].update({"path": str(output_path), "clipped_samples": written["clipped_samples"]})
    if residual_path is not None and result.residual is not None:
        save(residual_path, result.residual, result.sample_rate, settings.output_subtype)
        report["output"]["residual_path"] = str(residual_path)
    report["timing_s"]["total"] = round(time.perf_counter() - t0, 3)
    return report
