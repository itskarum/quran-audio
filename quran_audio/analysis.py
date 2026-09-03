"""Signal analysis: level, clipping, noise floor, pauses, bandwidth, hum.

Produces an `Analysis` value that the pipeline uses to decide what to do
and how hard. Nothing in this module modifies audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio_io import resample
from .stft import STFT, frame_length_for

_EPS = 1e-20


@dataclass
class HumHarmonic:
    freq_hz: float
    prominence_db: float
    width_hz: float


@dataclass
class HumReport:
    detected: bool
    fundamental_hz: float
    harmonics: list[HumHarmonic]
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "fundamental_hz": round(self.fundamental_hz, 3),
            "harmonics": [
                {"freq_hz": round(h.freq_hz, 3), "prominence_db": round(h.prominence_db, 1),
                 "width_hz": round(h.width_hz, 2)} for h in self.harmonics],
            "note": self.note,
        }


@dataclass
class Analysis:
    sample_rate: int
    n_samples: int
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    dc_offset: float
    clipped_runs: int
    noise_floor_dbfs: float
    snr_db: float
    speech_fraction: float
    pause_fraction: float
    bandwidth_hz: float
    low_edge_hz: float
    hum: HumReport
    n_fft: int = 0
    hop: int = 0
    pause_ranges: list[tuple[int, int]] = field(default_factory=list, repr=False)
    psd_freqs: np.ndarray | None = field(default=None, repr=False)
    noise_psd: np.ndarray | None = field(default=None, repr=False)
    speech_psd: np.ndarray | None = field(default=None, repr=False)
    # per-pause noise spectra: frame index of each pause centre and its mean
    # power spectrum, so the denoiser can follow slow drifts in the noise
    pause_frames: np.ndarray | None = field(default=None, repr=False)
    pause_psds: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "duration_s": round(self.duration_s, 3),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "dc_offset": round(self.dc_offset, 6),
            "clipped_runs": self.clipped_runs,
            "noise_floor_dbfs": round(self.noise_floor_dbfs, 2),
            "snr_db": round(self.snr_db, 2),
            "speech_fraction": round(self.speech_fraction, 3),
            "pause_fraction": round(self.pause_fraction, 3),
            "pause_count": len(self.pause_ranges),
            "bandwidth_hz": round(self.bandwidth_hz, 1),
            "low_edge_hz": round(self.low_edge_hz, 1),
            "hum": self.hum.to_dict(),
        }


def db(power: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(power, _EPS))


def count_clipped_runs(x: np.ndarray, ratio: float = 0.999, min_run: int = 3) -> tuple[int, float]:
    """Runs of >= `min_run` consecutive samples sitting at the signal peak.
    Real audio almost never holds its exact maximum for several samples,
    so many such runs mean the transfer was clipped."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 0:
        return 0, peak
    flat = np.abs(x) >= ratio * peak
    starts, ends = _runs(flat)
    lengths = ends - starts
    return int(np.count_nonzero(lengths >= min_run)), peak


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) indices of True runs."""
    if not mask.any():
        return np.zeros(0, int), np.zeros(0, int)
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _smooth_log_freq(psd: np.ndarray, freqs: np.ndarray, octaves: float = 1.0 / 6) -> np.ndarray:
    """Average each bin over a +-octaves/2 window (constant-Q smoothing)."""
    out = np.empty_like(psd)
    half = 2.0 ** (octaves / 2)
    for i, f in enumerate(freqs):
        lo = np.searchsorted(freqs, f / half)
        hi = np.searchsorted(freqs, f * half, side="right")
        out[i] = np.mean(psd[lo:max(hi, lo + 1)])
    return out


def analyze(x: np.ndarray, sr: int, hum_search: bool = True) -> Analysis:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    st = STFT(frame_length_for(sr))
    freqs = st.freqs(sr)
    band = (freqs >= 100.0) & (freqs <= min(4000.0, 0.45 * sr))
    n_frames = st.n_frames(n)

    # pass 1: per-frame in-band energy
    band_energy = np.empty(n_frames)
    for m0, power in st.power_blocks(x):
        band_energy[m0:m0 + power.shape[0]] = power[:, band].mean(axis=1)
    e_db = db(band_energy)
    floor_seed = float(np.percentile(e_db, 5))
    near_floor = e_db[e_db < floor_seed + 6.0]
    noise_floor = float(np.median(near_floor)) if near_floor.size else floor_seed
    noise_mask = e_db < noise_floor + 4.5
    speech_mask = e_db > noise_floor + 10.0
    if not speech_mask.any():             # very low SNR or silence: take the top quarter
        speech_mask = e_db >= np.percentile(e_db, 75)

    # pass 2: mean spectra of confident noise / speech frames, plus one
    # spectrum per pause (run of noise frames) for time-varying profiles
    run_starts, run_ends = _runs(noise_mask)
    run_id = np.full(n_frames, -1, dtype=np.int64)
    for k, (s0, e0) in enumerate(zip(run_starts, run_ends)):
        run_id[s0:e0] = k
    run_acc = np.zeros((len(run_starts), st.n_bins))
    noise_psd = np.zeros(st.n_bins)
    speech_psd = np.zeros(st.n_bins)
    n_noise = n_speech = 0
    for m0, power in st.power_blocks(x):
        m1 = m0 + power.shape[0]
        nm, sm = noise_mask[m0:m1], speech_mask[m0:m1]
        if nm.any():
            noise_psd += power[nm].sum(axis=0)
            n_noise += int(nm.sum())
            rid = run_id[m0:m1]
            for k in np.unique(rid[rid >= 0]):
                run_acc[k] += power[rid == k].sum(axis=0)
        if sm.any():
            speech_psd += power[sm].sum(axis=0)
            n_speech += int(sm.sum())
    noise_psd /= max(n_noise, 1)
    speech_psd /= max(n_speech, 1)

    snr = db(max(speech_psd[band].mean() - noise_psd[band].mean(), _EPS)) - db(noise_psd[band].mean())
    bandwidth, low_edge = _band_edges(speech_psd, noise_psd, freqs, sr)
    pause_ranges, pause_frames, pause_psds = _pauses(run_starts, run_ends, run_acc, st, n, sr)

    clipped_runs, peak = count_clipped_runs(x)
    rms = float(np.sqrt(np.mean(x * x))) if n else 0.0
    # frame RMS floor in dBFS: band energy is a mean |X|^2 per bin of a
    # sqrt-Hann-windowed frame; convert to an RMS-equivalent level.
    window_power = float(np.mean(st.window ** 2))
    floor_dbfs = noise_floor - db(st.n_fft * window_power * 0.5) if np.isfinite(noise_floor) else -120.0

    hum = detect_hum(x, sr, pause_ranges) if hum_search else HumReport(False, 0.0, [], "skipped")

    return Analysis(
        sample_rate=int(sr), n_samples=n, duration_s=n / sr,
        peak_dbfs=float(20 * np.log10(max(peak, 1e-9))),
        rms_dbfs=float(20 * np.log10(max(rms, 1e-9))),
        dc_offset=float(np.mean(x)) if n else 0.0,
        clipped_runs=clipped_runs,
        noise_floor_dbfs=float(floor_dbfs),
        snr_db=float(snr),
        speech_fraction=float(speech_mask.mean()),
        pause_fraction=float(noise_mask.mean()),
        bandwidth_hz=bandwidth, low_edge_hz=low_edge, hum=hum,
        n_fft=st.n_fft, hop=st.hop,
        pause_ranges=pause_ranges, psd_freqs=freqs, noise_psd=noise_psd, speech_psd=speech_psd,
        pause_frames=pause_frames, pause_psds=pause_psds,
    )


def _band_edges(speech_psd, noise_psd, freqs, sr, margin_db: float = 3.0) -> tuple[float, float]:
    """Highest and lowest frequency where speech frames beat the noise by
    `margin_db` after 1/6-octave smoothing. Above the high edge there is
    only noise, so a low-pass there costs nothing."""
    valid = freqs > 0
    excess = db(_smooth_log_freq(speech_psd[valid], freqs[valid])) - db(_smooth_log_freq(noise_psd[valid], freqs[valid]))
    f = freqs[valid]
    strong = excess >= margin_db
    nyq = sr / 2.0
    high = nyq
    start = np.searchsorted(f, 1000.0)
    # walk upward from 1 kHz; stop at the first 1/3-octave-wide hole
    i = start
    while i < len(f):
        if not strong[i]:
            j = np.searchsorted(f, f[i] * 2 ** (1 / 3))
            if not strong[i:j].any():
                high = float(f[i])
                break
        i += 1
    low = 0.0
    i = min(np.searchsorted(f, 300.0), len(f) - 1)
    while i >= 0:
        if not strong[i]:
            low = float(f[i])
            break
        i -= 1
    return float(min(high, nyq)), low


def _pauses(run_starts, run_ends, run_acc, st: STFT, n: int, sr: int, min_seconds: float = 0.3):
    """Sample ranges, centre frames and mean spectra of pauses at least
    `min_seconds` long."""
    ranges: list[tuple[int, int]] = []
    frames: list[float] = []
    psds: list[np.ndarray] = []
    centre = st.n_fft // 2 - (st.n_fft - st.hop)
    for k, (s0, e0) in enumerate(zip(run_starts, run_ends)):
        a = max(0, int(s0) * st.hop + centre)
        b = min(n, int(e0) * st.hop + centre)
        if b - a >= min_seconds * sr:
            ranges.append((int(a), int(b)))
            frames.append((int(s0) + int(e0) - 1) / 2.0)
            psds.append(run_acc[k] / max(int(e0) - int(s0), 1))
    if not frames:
        return ranges, None, None
    return ranges, np.asarray(frames), np.asarray(psds)


# ----- hum ----------------------------------------------------------------

_HUM_RATE = 4000            # analysis rate: keeps harmonics to 1.8 kHz, long windows cheap
_HUM_FMAX = 1800.0


def _median_psd(x: np.ndarray, sr: int, nperseg: int, max_segments: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch PSD (Hann, 2x zero padding) as (freqs, median, 10th percentile)
    across segments. The median keeps stationary tones (hum) and rejects
    what is only present some of the time (the voice); the 10th percentile
    is stricter still: a line must be there in nearly every segment."""
    n = len(x)
    nperseg = min(nperseg, n)
    nfft = 2 * nperseg
    n_seg_total = max(1, (n - nperseg) // (nperseg // 2) + 1)
    picks = np.unique(np.linspace(0, n_seg_total - 1, min(max_segments, n_seg_total)).astype(int))
    win = np.hanning(nperseg + 2)[1:-1]
    spectra = np.empty((len(picks), nfft // 2 + 1))
    for i, s in enumerate(picks):
        start = int(s * (nperseg // 2))
        seg = x[start:start + nperseg] * win
        spectra[i] = np.abs(np.fft.rfft(seg, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    return freqs, np.median(spectra, axis=0), np.percentile(spectra, 10, axis=0)


def _peak_near(freqs, psd, f, tol_hz):
    lo = np.searchsorted(freqs, f - tol_hz)
    hi = np.searchsorted(freqs, f + tol_hz, side="right")
    if hi <= lo:
        return f, 0.0
    k = lo + int(np.argmax(psd[lo:hi]))
    # parabolic interpolation in dB
    if 0 < k < len(psd) - 1:
        y0, y1, y2 = db(psd[k - 1]), db(psd[k]), db(psd[k + 1])
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
        delta = float(np.clip(delta, -1, 1))
    else:
        delta = 0.0
    return float(freqs[k] + delta * (freqs[1] - freqs[0])), float(psd[k])


def _prominence(freqs, psd, f, tol_hz):
    fpk, pk = _peak_near(freqs, psd, f, tol_hz)
    if pk <= 0:
        return fpk, 0.0
    sel = ((freqs >= f - 12) & (freqs <= f - 3)) | ((freqs >= f + 3) & (freqs <= f + 12))
    local = np.median(psd[sel]) if sel.any() else _EPS
    return fpk, float(db(pk) - db(local))


def _width(freqs, psd, fpk):
    k = int(np.argmin(np.abs(freqs - fpk)))
    half = psd[k] / 2.0
    lo = k
    while lo > 0 and psd[lo] > half:
        lo -= 1
    hi = k
    while hi < len(psd) - 1 and psd[hi] > half:
        hi += 1
    return max(float(freqs[hi] - freqs[lo]), 2.0 * float(freqs[1] - freqs[0]))


def detect_hum(x: np.ndarray, sr: int, pause_ranges: list[tuple[int, int]] | None = None,
               min_prominence_db: float = 6.0) -> HumReport:
    """Find mains hum (fundamental anywhere in 45-65 Hz, so speed-shifted
    transfers are caught) and its harmonics up to 1.8 kHz."""
    if len(x) < 2 * sr:
        return HumReport(False, 0.0, [], "signal too short for hum analysis")
    xl = resample(x, sr, _HUM_RATE) if sr != _HUM_RATE else x
    freqs, psd, persistent = _median_psd(xl, _HUM_RATE, nperseg=min(1 << 15, len(xl) // 2))
    tol = max(0.6, 2.5 * (freqs[1] - freqs[0]))

    best_f0, best_score = 0.0, -1.0
    for f0 in np.arange(45.0, 65.01, 0.1):
        score = 0.0
        for k in range(1, 9):
            _, prom = _prominence(freqs, psd, k * f0, tol)
            score += max(prom - 3.0, 0.0) / k
        if score > best_score:
            best_f0, best_score = float(f0), score

    # refine f0 from the measured harmonic peaks, then measure every harmonic
    est, weights = [], []
    for k in range(1, 5):
        fpk, prom = _prominence(freqs, psd, k * best_f0, tol)
        if prom >= min_prominence_db:
            est.append(fpk / k)
            weights.append(prom)
    f0 = float(np.average(est, weights=weights)) if est else best_f0

    # Confirmation material: hum is there whether or not anyone is speaking,
    # a sustained voice harmonic is not. With enough pause material every
    # line must show up there too, and a line that the voice masks in the
    # whole-file spectrum can still be found in the pauses.
    pause_psd = _pause_psd(xl, sr, pause_ranges)
    harmonics: list[HumHarmonic] = []
    k = 1
    while k * f0 <= _HUM_FMAX:
        target = k * f0
        fpk, prom = _prominence(freqs, psd, target, tol + 0.02 * k)
        persistent_ok = _prominence(freqs, persistent, fpk, tol)[1] >= 0.75 * min_prominence_db
        whole_ok = prom >= min_prominence_db and persistent_ok
        if pause_psd is not None:
            pf, pp, _ = pause_psd
            ptol = max(3.0, 2.5 * (pf[1] - pf[0]))
            ppk, pprom = _prominence(pf, pp, target, ptol)
            if pprom >= min_prominence_db:
                if not whole_ok:
                    # masked by the voice in the whole-file spectrum: locate the
                    # line precisely near the pause estimate
                    fpk, _ = _peak_near(freqs, psd, ppk, 1.5)
                    prom = pprom
                harmonics.append(HumHarmonic(fpk, prom, _width(freqs, psd, fpk)))
        elif whole_ok and prom >= 10.0 and k <= 3:
            # nothing to confirm against: only unmistakable low harmonics, and
            # (below) a strong fundamental itself, since a voice has no energy
            # at 45-65 Hz but its drifting harmonics can fake a comb above it
            harmonics.append(HumHarmonic(fpk, prom, _width(freqs, psd, fpk)))
        k += 1
    note = "harmonics confirmed in pauses" if pause_psd is not None else "no usable pauses; only strong low harmonics kept"
    prom_by_k = {int(round(h.freq_hz / f0)): h.prominence_db for h in harmonics}
    p1, p2 = prom_by_k.get(1, 0.0), prom_by_k.get(2, 0.0)
    if pause_psd is not None:
        detected = (max(p1, p2) >= 8.0 and len(harmonics) >= 2) or p1 >= 12.0
    else:
        detected = p1 >= 12.0 and (len(harmonics) >= 2 or p1 >= 18.0)
    if not detected:
        note = "no stationary comb above threshold"
    return HumReport(detected, f0 if detected else 0.0, harmonics if detected else [], note)


def _pause_psd(xl: np.ndarray, sr: int, pause_ranges, min_total_s: float = 2.0):
    """Median PSD of the pause material (at the hum analysis rate), or None
    when there is less than `min_total_s` of it."""
    if not pause_ranges:
        return None
    pieces = [xl[int(a * _HUM_RATE / sr):int(b * _HUM_RATE / sr)] for a, b in pause_ranges
              if (b - a) >= 0.5 * sr]
    total = sum(len(p) for p in pieces)
    if total < min_total_s * _HUM_RATE:
        return None
    tapered = []
    for p in pieces:
        w = np.ones(len(p))
        edge = max(1, min(len(p) // 4, _HUM_RATE // 100))
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
        w[:edge] = ramp
        w[-edge:] = ramp[::-1]
        tapered.append(p * w)
    return _median_psd(np.concatenate(tapered), _HUM_RATE, nperseg=1 << 11, max_segments=512)
