#!/usr/bin/env python
"""Objective evaluation: degrade a clean reference the way an old transfer
does, restore it with each preset/backend, and measure how close the result
is to the reference.

    python tools/evaluate.py REFERENCE.wav [--snr 14] [--backends classical,dfn]
        [--presets gentle,standard,strong] [--out DIR] [--seconds 60] [--start 5]
        [--rt60 1.2] [--drr 0] [--no-flutter] [--no-breaths] [--mp3 128]
        [--noise-file TRANSFER.wav] [--bandwidth 4500] [--no-eq] [--seed 5]

The reference is the performance as it was in the room: the clean voice
convolved with a hall (synthetic, RT60 --rt60 s, direct-to-reverberant
ratio --drr dB), breaths added in its pauses, band-limited to --bandwidth,
and carrying the transfer's wow and flutter. None of that is the tool's to
remove, so it sits in the reference and counts as fidelity, not as error.

The degradation is the transfer: hiss (pink + white, or noise taken from
the pauses of a real transfer with --noise-file) at --snr, a 50 Hz mains
comb that shares the tape's flutter, 400 clicks, +-3 dB slow level drift,
and an MP3 round trip at --mp3 kbps (pre-echo, spectral holes); 0 skips it.

Metrics (higher is better unless noted):
  SI-SDR   scale-invariant signal-to-distortion ratio vs the reference, dB
  STOI     short-time objective intelligibility (needs the `eval` extra), 0..1
  coh      mean magnitude-squared coherence with the reference over 150-3500 Hz:
           insensitive to EQ and level, sensitive to noise and speech distortion
  pause    residual level in the reference's pauses, dBFS (lower is better)
  clicks   fraction of injected clicks whose error fell by at least 10 dB
  hum      attenuation of the 50/100/150 Hz lines in the pauses, dB
  voice    voice-band (300-3000 Hz) level change in speech frames vs the reference, dB
  tail     extra cut of the room decay 300 ms after phrase ends vs the reference, dB
  breath   level change of the inserted breaths vs the reference, dB (0 = kept)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, coherence, fftconvolve, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quran_audio import analysis, audio_io, enhance_signal, fidelity, make_settings  # noqa: E402

HUM_HZ = (50.0, 100.0, 150.0)


def si_sdr(ref, est):
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    s = a * ref
    return float(10 * np.log10(np.sum(s ** 2) / (np.sum((est - s) ** 2) + 1e-20)))


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    X[1:] /= np.sqrt(np.maximum(f[1:], f[1]) / f[1])
    X[0] = 0
    y = np.fft.irfft(X, n)
    return y / np.std(y)


# ----- the room ---------------------------------------------------------
def hall_rir(sr: int, rt60: float, drr_db: float, rng) -> np.ndarray:
    """Synthetic hall: direct path, six early reflections in the first
    40 ms, then a diffuse tail whose decay is longer at low frequencies
    (RT60 x1.3 below 500 Hz, x1 to 2 kHz, x0.7 to 4 kHz, x0.5 above), as
    in a plastered hall with carpet and people."""
    n = int(sr * rt60 * 1.3)
    t = np.arange(n) / sr
    tail = np.zeros(n)
    edges = [(20.0, 500.0, 1.3), (500.0, 2000.0, 1.0), (2000.0, 4000.0, 0.7), (4000.0, 0.45 * sr, 0.5)]
    for lo, hi, k in edges:
        if lo >= 0.45 * sr:
            continue
        sos = butter(4, [lo, min(hi, 0.45 * sr)], btype="band", fs=sr, output="sos")
        band = sosfiltfilt(sos, rng.standard_normal(n))
        tail += band * np.exp(-6.908 * t / (rt60 * k))
    tail[: int(0.005 * sr)] = 0.0               # nothing diffuse before 5 ms
    early = np.zeros(n)
    for d in rng.uniform(0.005, 0.04, 6):
        early[int(d * sr)] += rng.uniform(0.25, 0.6) * rng.choice([-1.0, 1.0])
    rev = early + tail * (np.sqrt(np.sum(early ** 2)) / np.sqrt(np.sum(tail ** 2)) * 1.5)
    rev *= 10 ** (-drr_db / 20) / np.sqrt(np.sum(rev ** 2))     # direct energy 1, reverberant energy 10^(-drr/10)
    rir = rev
    rir[0] += 1.0
    return rir


def add_room(x: np.ndarray, sr: int, rt60: float, drr_db: float, rng) -> np.ndarray:
    if rt60 <= 0:
        return x
    return fftconvolve(x, hall_rir(sr, rt60, drr_db, rng))[: len(x)]


# ----- breaths ----------------------------------------------------------
def add_breaths(x: np.ndarray, sr: int, pauses: list[tuple[int, int]], rng, level_db: float = -30.0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """A 250-400 ms inhalation (500 Hz-4 kHz noise, quick rise, slower
    fall) in every pause longer than 0.7 s, `level_db` below the speech
    RMS. Returns the signal and the breath sample ranges."""
    y = x.copy()
    levels = fidelity_levels(x, sr)
    speech_rms = 10 ** (levels / 20)
    sos = butter(4, [500.0, min(4000.0, 0.45 * sr)], btype="band", fs=sr, output="sos")
    out = []
    for p0, p1 in pauses:
        if (p1 - p0) / sr < 0.7:
            continue
        dur = rng.uniform(0.25, 0.4)
        a = p0 + int(0.15 * sr)
        b = min(a + int(dur * sr), p1 - int(0.1 * sr))
        if b - a < int(0.2 * sr):
            continue
        n = b - a
        burst = sosfiltfilt(sos, rng.standard_normal(n + 2 * sr // 10))[sr // 10: sr // 10 + n]
        rise, fall = int(0.35 * n), n - int(0.35 * n)
        env = np.concatenate([np.sin(np.linspace(0, np.pi / 2, rise)) ** 2, np.cos(np.linspace(0, np.pi / 2, fall)) ** 2])
        burst = burst / (np.std(burst) + 1e-12) * speech_rms * 10 ** (level_db / 20) * env
        y[a:b] += burst
        out.append((a, b))
    return y, out


def fidelity_levels(x: np.ndarray, sr: int) -> float:
    """Speech RMS in dBFS: RMS of the 50 ms frames above the floor + 15 dB."""
    n = int(0.05 * sr)
    frames = x[: len(x) // n * n].reshape(-1, n)
    lv = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)
    seed = float(np.percentile(lv, 5))
    floor = float(np.median(lv[lv < seed + 6]))
    sp = lv[lv > floor + 15]
    return float(np.median(sp)) if sp.size else float(np.percentile(lv, 90))


# ----- the transfer -----------------------------------------------------
def flutter_warp(x: np.ndarray, sr: int, rng, wow_pct: float = 0.2, wow_hz: float = 0.6,
                 flutter_pct: float = 0.1, flutter_hz: float = 7.0) -> np.ndarray:
    """Read `x` at a speed that wanders: a slow wow and a faster flutter,
    as a worn capstan or a stretched tape do. The same warp goes on the
    reference and on the hum, so nothing here counts as error."""
    n = len(x)
    t = np.arange(n) / sr
    speed = (1.0 + wow_pct / 100 * np.sin(2 * np.pi * wow_hz * t + rng.uniform(0, 2 * np.pi))
             + flutter_pct / 100 * np.sin(2 * np.pi * flutter_hz * t + rng.uniform(0, 2 * np.pi)))
    pos = np.cumsum(speed) - speed[0]
    pos *= (n - 1) / pos[-1]                      # same length in and out
    return np.interp(pos, np.arange(n), x)


def transfer_noise(n: int, sr: int, rng, noise_file: str | None) -> np.ndarray:
    """Unit-RMS noise: pink + white, or the pauses of a real transfer."""
    if not noise_file:
        y = 0.7 * pink(n, rng) + 0.7 * rng.standard_normal(n)
        return y / np.std(y)
    a = audio_io.load(noise_file)
    m = audio_io.to_mono(a.samples, sr=a.sample_rate)[0]
    if a.sample_rate != sr:
        m = audio_io.resample(m, a.sample_rate, sr)
    m = m - m.mean()
    pauses = analysis.analyze(m, sr, hum_search=False).pause_ranges
    pieces = [m[p0 + int(0.2 * sr): p1] for p0, p1 in pauses if p1 - p0 >= int(0.5 * sr)]
    if not pieces or sum(len(p) for p in pieces) < sr:
        pieces = [m]                              # a pure noise recording
    fade = int(0.02 * sr)
    ramp = np.linspace(0, 1, fade)
    out = np.zeros(n + fade)
    pos = 0
    while pos < n:
        piece = pieces[rng.integers(len(pieces))].copy()
        if len(piece) <= 2 * fade:
            continue
        piece[:fade] *= ramp
        piece[-fade:] *= ramp[::-1]
        end = min(pos + len(piece), n + fade)
        out[pos:end] += piece[: end - pos]
        pos += len(piece) - fade
    out = out[:n]
    return out / (np.std(out) + 1e-12)


def mp3_round_trip(x: np.ndarray, sr: int, kbps: int) -> np.ndarray:
    """Encode to MP3 at a constant bitrate and decode again, re-aligned to
    the input to the sample."""
    with tempfile.TemporaryDirectory(prefix="quran-audio-eval-") as tmp:
        path = Path(tmp) / "roundtrip.mp3"
        audio_io.save(path, x, sr, mp3_kbps=float(kbps))
        y = audio_io.to_mono(audio_io.load(path).samples, sr=sr)[0]
    n = len(x)
    c0, c1 = n // 2 - min(n // 4, 5 * sr), n // 2 + min(n // 4, 5 * sr)
    seg = x[c0:c1]
    best, best_lag = -np.inf, 0
    for lag in range(-4096, 4097):
        a, b = c0 + lag, c1 + lag
        if a < 0 or b > len(y):
            continue
        c = float(np.dot(seg, y[a:b]))
        if c > best:
            best, best_lag = c, lag
    out = np.zeros(n)
    src = y[max(0, best_lag): max(0, best_lag) + n]
    dst0 = max(0, -best_lag)
    m = min(len(src), n - dst0)
    out[dst0: dst0 + m] = src[:m]
    return out


def degrade(clean: np.ndarray, sr: int, args, rng) -> dict:
    clean = clean / (np.max(np.abs(clean)) + 1e-12) * 0.5
    room = add_room(clean, sr, args.rt60, args.drr, rng)
    if args.bandwidth > 0:
        room = sosfiltfilt(butter(6, min(args.bandwidth, 0.45 * sr), fs=sr, output="sos"), room)
    pauses = analysis.analyze(room, sr, hum_search=False).pause_ranges
    breaths: list[tuple[int, int]] = []
    if not args.no_breaths:
        room, breaths = add_breaths(room, sr, pauses, rng)
    t = np.arange(len(room)) / sr
    hum = 0.01 * np.sin(2 * np.pi * 50 * t) + 0.004 * np.sin(2 * np.pi * 100 * t + 1) + 0.003 * np.sin(2 * np.pi * 150 * t + 2)
    if args.no_flutter:
        ref, taped = room, room + hum
    else:
        ref = flutter_warp(room, sr, np.random.default_rng(args.seed + 1))
        taped = flutter_warp(room + hum, sr, np.random.default_rng(args.seed + 1))
    noise = transfer_noise(len(ref), sr, rng, args.noise_file)
    noise *= np.sqrt(np.sum(ref ** 2) / len(ref)) * 10 ** (-args.snr / 20)
    clicks = np.zeros_like(ref)
    pos = rng.integers(1000, len(ref) - 1000, 400)
    for p in pos:
        clicks[p:p + rng.integers(1, 6)] += rng.uniform(0.05, 0.4) * rng.choice([-1, 1])
    drift = 10 ** (np.sin(2 * np.pi * 0.02 * t) * 3 / 20)
    degraded = (taped + noise + hum * 0 + clicks) * drift
    scale = 0.5 / np.max(np.abs(degraded))
    ref, degraded = ref * scale, degraded * scale
    if args.mp3 > 0:
        degraded = mp3_round_trip(degraded, sr, args.mp3)
    ref_pauses = analysis.analyze(ref, sr, hum_search=False).pause_ranges
    return {"ref": ref, "degraded": degraded, "clicks": pos, "breaths": breaths, "pauses": ref_pauses}


# ----- metrics ----------------------------------------------------------
def hum_prominence_db(x: np.ndarray, sr: int, pz: np.ndarray) -> float:
    """Mean prominence of the 50/100/150 Hz lines over the pause samples."""
    seg = x[pz]
    if len(seg) < 4 * sr:
        return float("nan")
    f, p = welch(seg, fs=sr, nperseg=4 * sr, noverlap=2 * sr)
    vals = []
    for h in HUM_HZ:
        line = p[(f >= h - 1.0) & (f <= h + 1.0)].max()
        around = np.median(p[(f >= h - 12.0) & (f <= h + 12.0) & ((f < h - 2.0) | (f > h + 2.0))])
        vals.append(10 * np.log10(line / max(around, 1e-30)))
    return float(np.mean(vals))


def breath_change_db(ref: np.ndarray, y: np.ndarray, sr: int, breaths: list[tuple[int, int]], pauses: list[tuple[int, int]]) -> float | None:
    """Level of the inserted breaths in `y` above `y`'s own floor in the
    same pause, relative to the breath level in the reference."""
    vals = []
    for a, b in breaths:
        host = next(((p0, p1) for p0, p1 in pauses if p0 <= a and b <= p1), None)
        e_ref = np.mean(ref[a:b] ** 2)
        e_y = np.mean(y[a:b] ** 2)
        if host is not None:
            rest = np.concatenate([y[host[0] + int(0.15 * sr): a], y[b: host[1]]])
            floor = np.mean(rest ** 2) if rest.size > int(0.1 * sr) else 0.0
            e_y = max(e_y - floor, e_ref * 1e-4)
        vals.append(10 * np.log10(e_y / max(e_ref, 1e-30)))
    return round(float(np.median(vals)), 2) if len(vals) >= 3 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("--snr", type=float, default=14.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--start", type=float, default=5.0)
    ap.add_argument("--backends", default="classical")
    ap.add_argument("--presets", default="gentle,standard,strong")
    ap.add_argument("--out", help="directory for reference/degraded/restored files and results.json")
    ap.add_argument("--no-eq", action="store_true", help="disable high-pass, low-pass and tonal balance")
    ap.add_argument("--rt60", type=float, default=1.2, help="hall reverberation time in s (0 = dry)")
    ap.add_argument("--drr", type=float, default=0.0, help="direct-to-reverberant ratio in dB")
    ap.add_argument("--no-flutter", action="store_true", help="steady transfer speed")
    ap.add_argument("--no-breaths", action="store_true", help="no inhalations in the pauses")
    ap.add_argument("--mp3", type=int, default=128, help="MP3 round trip at this bitrate (0 = none)")
    ap.add_argument("--noise-file", help="a real transfer: its pauses supply the hiss instead of synthetic noise")
    ap.add_argument("--bandwidth", type=float, default=4500.0, help="band-limit of the reference in Hz (0 = full)")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    a = audio_io.load(args.reference)
    mono = audio_io.to_mono(a.samples, sr=a.sample_rate)[0]
    sr = a.sample_rate
    clean = mono[int(args.start * sr):int((args.start + args.seconds) * sr)]
    clean = clean - clean.mean()
    d = degrade(clean, sr, args, rng)
    ref, degraded, click_pos, breaths, pauses = d["ref"], d["degraded"], d["clicks"], d["breaths"], d["pauses"]
    pz = np.concatenate([np.arange(s, e) for s, e in pauses]) if pauses else np.arange(0, sr)
    hum_before = hum_prominence_db(degraded, sr, pz)
    try:
        from pystoi import stoi
    except ImportError:
        stoi = None

    def band_coherence(y):
        f, c = coherence(ref, y, fs=sr, nperseg=2048)
        sel = (f >= 150) & (f <= 3500)
        return float(np.mean(c[sel]))

    def metrics(y):
        err_b = np.array([np.max(np.abs((degraded - ref)[p:p + 8])) for p in click_pos])
        err_a = np.array([np.max(np.abs((y - ref)[p:p + 8])) for p in click_pos])
        fid = fidelity.measure(ref, y, sr)
        m = {"si_sdr_db": round(si_sdr(ref, y), 2), "coherence": round(band_coherence(y), 3),
             "pause_dbfs": round(float(20 * np.log10(np.std(y[pz]) + 1e-12)), 1),
             "clicks_fixed": round(float(np.mean(err_a < err_b * 10 ** (-10 / 20))), 2),
             "hum_attenuation_db": round(hum_before - hum_prominence_db(y, sr, pz), 1),
             "voice_band_db": fid["voice_band_retention_db"],
             "onset_db": fid["onset_retention_db"],
             "tail_db": None if not fid["tail_300ms_db"] else fid["tail_300ms_db"]["extra_cut"],
             "breath_db": breath_change_db(ref, y, sr, breaths, pauses)}
        if stoi is not None:
            m["stoi"] = round(float(stoi(ref, y, sr)), 3)
        return m

    degradation = {"snr_db": args.snr, "rt60_s": args.rt60, "drr_db": args.drr, "flutter": not args.no_flutter,
                   "breaths": len(breaths), "mp3_kbps": args.mp3, "noise": args.noise_file or "pink+white",
                   "bandwidth_hz": args.bandwidth, "clicks": int(len(click_pos)), "seed": args.seed}
    results = {"reference": str(args.reference), "sample_rate": sr, "seconds": len(ref) / sr,
               "degradation": degradation, "degraded": metrics(degraded), "runs": []}
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        audio_io.save(out / "reference.wav", ref, sr)
        audio_io.save(out / "degraded.wav", degraded, sr)
    print(f"reference {args.reference}: {len(ref)/sr:.0f} s at {sr} Hz; room RT60 {args.rt60} s at DRR {args.drr} dB, "
          f"{len(breaths)} breaths, flutter {'off' if args.no_flutter else 'on'}; transfer: {degradation['noise']} hiss at "
          f"{args.snr} dB SNR, 50 Hz comb ({hum_before:.0f} dB above the hiss), {len(click_pos)} clicks, drift, "
          f"MP3 {args.mp3 or 'none'} kbps")
    cols = f"{'run':24s} {'SI-SDR':>7s} {'STOI':>6s} {'coh':>6s} {'pause':>7s} {'clicks':>6s} {'hum':>6s} {'voice':>6s} {'tail':>6s} {'breath':>6s} {'time':>6s}"
    print(cols)

    def fmt(v, w, p):
        return f"{'n/a':>{w}s}" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:{w}.{p}f}"

    def row(name, m, dt=None):
        print(f"{name:24s} {m['si_sdr_db']:7.2f} {fmt(m.get('stoi'), 6, 3)} {m['coherence']:6.3f} {m['pause_dbfs']:7.1f} "
              f"{m['clicks_fixed']:6.2f} {fmt(m['hum_attenuation_db'], 6, 1)} {fmt(m['voice_band_db'], 6, 2)} "
              f"{fmt(m['tail_db'], 6, 1)} {fmt(m['breath_db'], 6, 1)} {'' if dt is None else f'{dt:6.1f}'}")

    row("degraded", results["degraded"])
    for backend in args.backends.split(","):
        for preset in args.presets.split(","):
            overrides = {"denoise": backend, "target_lufs": None, "leveler": False}
            if args.no_eq:
                overrides.update({"highpass": False, "lowpass": False, "tonal_balance": False})
            settings = make_settings(preset, overrides)
            t0 = time.perf_counter()
            res = enhance_signal(degraded, sr, settings)
            dt = time.perf_counter() - t0
            y = res.samples[:, 0]
            m = metrics(y)
            m.update({"backend": res.report["denoise_backend"], "preset": preset, "seconds": round(dt, 1),
                      "fidelity_warnings": res.report.get("fidelity", {}).get("warnings", [])})
            results["runs"].append(m)
            row(f"{backend} {preset}", m, dt)
            if args.out:
                audio_io.save(Path(args.out) / f"restored_{backend}_{preset}.wav", y, sr)
    if args.out:
        (Path(args.out) / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
