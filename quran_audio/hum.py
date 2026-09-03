"""Mains-hum removal by tracking and subtracting each measured harmonic.

For every harmonic the analysis measured, the signal is demodulated at
that exact frequency, the resulting baseband is smoothed over a one to
two second window, and the reconstructed slowly varying sinusoid is
subtracted. A voice harmonic sweeping through the same frequency is far
too brief to move that estimate, so it passes untouched, and unlike a
narrow IIR notch there is no exponential ringing before or after speech
transients (a 1 Hz notch rings for half a second every time a harmonic
crosses it). Lines wider than a few hertz (unstable hum) fall back to a
zero-phase notch of matching width. Never a blind comb at every multiple
of 50 Hz.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import filtfilt, iirnotch

from .analysis import HumReport

# filtfilt squares the magnitude response, which pushes the -3 dB points
# outward by this factor for a second-order notch; compensate at design.
_FILTFILT_WIDENING = 1.554
_FINE_RATE = 400.0             # Hz; boxcar means before the Hann decimation
_CONTROL_RATE = 100.0          # Hz; the demodulated hum is well under 1 Hz wide
_NOTCH_ABOVE_WIDTH_HZ = 6.0    # wider lines are drifting: notch them instead


def tracker_window_s(width_hz: float) -> float:
    """Median window (in seconds of pause material) from the measured line
    width: narrow, stable lines get six seconds; wider lines are followed
    faster."""
    return float(np.clip(1.5 / max(width_hz, 0.1), 2.0, 6.0))


def _phasors(start: int, length: int, freqs: np.ndarray, sr: int) -> np.ndarray:
    """exp(-2j pi f t) for t = start..start+length-1, shape (len(freqs), length).
    Built by cumulative rotation per block, re-anchored exactly at `start`."""
    r = np.exp(-2j * np.pi * freqs / sr)
    steps = np.cumprod(np.broadcast_to(r[:, None], (len(freqs), length)), axis=1)
    anchor = np.exp(-2j * np.pi * freqs * (start - 1) / sr)
    return anchor[:, None] * steps


def _control_mask(n_ctrl: int, dec: int, pause_ranges) -> np.ndarray:
    """True for control samples that lie inside a pause. Falls back to all
    True when there is too little pause material to estimate from."""
    if not pause_ranges:
        return np.ones(n_ctrl, dtype=bool)
    mask = np.zeros(n_ctrl, dtype=bool)
    for a, b in pause_ranges:
        mask[-(-a // dec):max(-(-a // dec), b // dec)] = True
    return mask if mask.sum() >= 2 * _CONTROL_RATE else np.ones(n_ctrl, dtype=bool)


def _hann_decimator(freqs: np.ndarray, comb_hz: float | None) -> np.ndarray:
    """Hann window (in fine boxes) whose length is two periods of the comb
    spacing, so every other harmonic of the same comb falls on a zero of
    the window's response instead of leaking into this line's baseband."""
    spacing = comb_hz if comb_hz else (float(np.min(np.diff(np.sort(freqs)))) if len(freqs) > 1 else float(freqs[0]))
    length = max(4, int(round(2.0 * _FINE_RATE / max(spacing, 1.0))))
    w = np.hanning(length + 2)[1:-1]
    return w / w.sum()


def track_and_subtract(x: np.ndarray, sr: int, freqs: np.ndarray, windows_s: np.ndarray,
                       pause_ranges=None, comb_hz: float | None = None,
                       block: int = 1 << 20) -> tuple[np.ndarray, np.ndarray]:
    """Remove the stationary sinusoid at each of `freqs` from `x`.

    The hum's complex amplitude is measured only inside the pauses (where
    there is no voice to confuse it with), its exact frequency is refined
    from the phase drift across those pauses, and the estimate is
    interpolated across the speech in between. Returns (cleaned,
    refined_freqs)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    freqs = np.array(freqs, dtype=np.float64)
    k = len(freqs)
    if k == 0 or n == 0:
        return x, freqs
    dec_f = max(1, int(round(sr / _FINE_RATE)))
    stride = 4                                  # control rate = fine rate / 4
    dec = dec_f * stride
    block = max(dec_f, (block // dec_f) * dec_f)
    n_fine = -(-n // dec_f)
    fine = np.zeros((k, n_fine), dtype=np.complex128)
    # pass 1: fine-rate boxcar means of the demodulated signal
    for start in range(0, n, block):
        end = min(n, start + block)
        prod = x[start:end][None, :] * _phasors(start, end - start, freqs, sr)
        m = end - start
        full = (m // dec_f) * dec_f
        i0 = start // dec_f
        if full:
            fine[:, i0:i0 + full // dec_f] += prod[:, :full].reshape(k, -1, dec_f).sum(axis=2)
        if full < m:
            fine[:, i0 + full // dec_f] += prod[:, full:].sum(axis=1)
    counts = np.full(n_fine, dec_f, dtype=np.float64)
    counts[-1] = n - (n_fine - 1) * dec_f
    fine /= counts[None, :]
    # Hann-weighted decimation to the control rate
    w = _hann_decimator(freqs, comb_hz)
    offset = stride // 2
    z = np.stack([np.convolve(fine[j], w, mode="same")[offset::stride] for j in range(k)])
    n_ctrl = z.shape[1]
    centres = (np.arange(n_ctrl) * stride + offset + 0.5) * dec_f - 0.5   # in samples
    t_ctrl = centres / sr

    valid = _control_mask(n_ctrl, dec, pause_ranges)
    idx_valid = np.flatnonzero(valid)
    both = valid[1:] & valid[:-1]
    refined = freqs.copy()
    smooth = np.empty_like(z)
    for j in range(k):
        win = max(3, int(round(windows_s[j] * _CONTROL_RATE)) | 1)

        def estimate(zj):
            series = zj[idx_valid]
            size = min(win, len(series))
            re = median_filter(series.real, size=size, mode="nearest")
            im = median_filter(series.imag, size=size, mode="nearest")
            strength = float(np.median(np.hypot(re, im)))
            full_re = np.interp(np.arange(n_ctrl), idx_valid, re)
            full_im = np.interp(np.arange(n_ctrl), idx_valid, im)
            return strength, full_re + 1j * full_im

        best_strength, smooth[j] = estimate(z[j])
        # frequency refinement: median phase advance per control step inside
        # pauses; accepted only if it yields a stronger stationary estimate
        if both.sum() >= 50:
            inc = np.angle(z[j, 1:] * np.conj(z[j, :-1]))[both]
            delta_hz = float(np.median(inc)) * _CONTROL_RATE / (2.0 * np.pi)
            if 0 < abs(delta_hz) < 1.0:
                strength, cand = estimate(z[j] * np.exp(-2j * np.pi * delta_hz * t_ctrl))
                if strength > best_strength:
                    best_strength, smooth[j], refined[j] = strength, cand, freqs[j] + delta_hz

    # pass 2: rebuild the hum at full rate and subtract it
    y = x.copy()
    for start in range(0, n, block):
        end = min(n, start + block)
        idx = np.arange(start, end, dtype=np.float64)
        ph = np.conj(_phasors(start, end - start, refined, sr))     # exp(+2j pi f t)
        est = np.zeros(end - start)
        for j in range(k):
            zi = np.interp(idx, centres, smooth[j].real) + 1j * np.interp(idx, centres, smooth[j].imag)
            est += 2.0 * (zi * ph[j]).real
        y[start:end] -= est
    return y, refined


def notch(x: np.ndarray, sr: int, freq_hz: float, bandwidth_hz: float) -> np.ndarray:
    bw_design = bandwidth_hz / _FILTFILT_WIDENING
    b, a = iirnotch(freq_hz, freq_hz / bw_design, fs=sr)
    padlen = int(min(len(x) - 1, 3 * sr / bw_design))
    return filtfilt(b, a, x, padlen=padlen)


def remove_hum(x: np.ndarray, sr: int, hum: HumReport, min_prominence_db: float = 6.0,
               max_harmonics: int = 24, pause_ranges=None) -> tuple[np.ndarray, list[dict]]:
    """Return (cleaned, applied) where `applied` lists every line handled.
    `pause_ranges` (sample ranges of breath pauses, from the analysis) is
    where the tracker measures the hum; without it, everything is used."""
    if not hum.detected:
        return x, []
    y = np.asarray(x, dtype=np.float64)
    applied: list[dict] = []
    tracked_f, tracked_w = [], []
    for h in hum.harmonics[:max_harmonics]:
        if h.prominence_db < min_prominence_db or h.freq_hz >= 0.45 * sr:
            continue
        if h.width_hz > _NOTCH_ABOVE_WIDTH_HZ:
            bw = float(np.clip(1.2 * h.width_hz, 1.0, 12.0))
            y = notch(y, sr, h.freq_hz, bw)
            applied.append({"freq_hz": round(h.freq_hz, 3), "method": "notch", "bandwidth_hz": round(bw, 2),
                            "prominence_db": round(h.prominence_db, 1)})
        else:
            w = tracker_window_s(h.width_hz)
            tracked_f.append(h.freq_hz)
            tracked_w.append(w)
            applied.append({"freq_hz": round(h.freq_hz, 3), "method": "track", "window_s": round(w, 2),
                            "prominence_db": round(h.prominence_db, 1)})
    if tracked_f:
        y, refined = track_and_subtract(y, sr, np.array(tracked_f), np.array(tracked_w), pause_ranges,
                                        comb_hz=hum.fundamental_hz)
        for entry, f in zip([a for a in applied if a["method"] == "track"], refined):
            entry["refined_hz"] = round(float(f), 3)
    return y, applied
