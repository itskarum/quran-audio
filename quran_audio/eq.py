"""Voice-band conditioning: rumble high-pass, bandwidth-aware low-pass and a
conservative tonal-balance correction toward the long-term average speech
spectrum. Every filter is zero-phase, so the output stays sample-aligned.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, firwin2, sosfiltfilt


def highpass(x: np.ndarray, sr: int, fc: float, order: int = 2) -> np.ndarray:
    if fc <= 0:
        return x
    sos = butter(order, fc, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, sr: int, fc: float, order: int = 4) -> np.ndarray:
    if fc >= 0.47 * sr:
        return x
    sos = butter(order, fc, btype="lowpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def rumble_cutoff(low_edge_hz: float) -> float:
    """Below 40 Hz there is never voice; above 80 Hz there may be a deep
    reciter's fundamental. Sit at 60 % of the measured low edge between."""
    return float(np.clip(0.6 * low_edge_hz, 40.0, 80.0))


def reference_ltass_db(freqs: np.ndarray) -> np.ndarray:
    """Smooth approximation of the universal long-term average speech
    spectrum of Byrne et al. (1994): rising about 3.9 dB/octave from 100 Hz
    to a plateau at 400-630 Hz, then falling about 4.8 dB/octave, and
    dropping steeply below 100 Hz. Only the *shape* is used; levels are
    normalised before comparison."""
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1.0)
    out = np.zeros_like(f)
    hi = f > 630.0
    out[hi] = -4.8 * np.log2(f[hi] / 630.0)
    lo = f < 400.0
    out[lo] = -3.9 * np.log2(400.0 / f[lo])
    very_low = f < 100.0
    out[very_low] -= 12.0 * np.log2(100.0 / f[very_low])
    return out


def third_octave_centres(f_lo: float, f_hi: float) -> np.ndarray:
    k = np.arange(np.ceil(3 * np.log2(f_lo / 1000.0)), np.floor(3 * np.log2(f_hi / 1000.0)) + 1)
    return 1000.0 * 2.0 ** (k / 3.0)


def band_levels_db(psd: np.ndarray, freqs: np.ndarray, centres: np.ndarray) -> np.ndarray:
    out = np.full(len(centres), np.nan)
    for i, c in enumerate(centres):
        sel = (freqs >= c / 2 ** (1.0 / 6)) & (freqs < c * 2 ** (1.0 / 6))
        if sel.any():
            out[i] = 10.0 * np.log10(np.mean(psd[sel]) + 1e-20)
    return out


def design_tonal_balance(speech_psd: np.ndarray, freqs: np.ndarray, sr: int, low_edge_hz: float,
                         bandwidth_hz: float, strength: float = 0.5, max_db: float = 6.0,
                         numtaps: int | None = None) -> tuple[np.ndarray | None, list[dict]]:
    """Linear-phase FIR that nudges the measured speech spectrum toward the
    reference shape. Corrections are smoothed over one octave (so formants
    are never carved), clamped to +-max_db, scaled by `strength`, never
    boost below 150 Hz and are zero outside the usable band."""
    f_lo = max(float(low_edge_hz), 100.0)
    f_hi = min(float(bandwidth_hz), 0.45 * sr)
    centres = third_octave_centres(f_lo, f_hi)
    if len(centres) < 4 or strength <= 0:
        return None, []
    meas = band_levels_db(speech_psd, freqs, centres)
    ok = np.isfinite(meas)
    centres, meas = centres[ok], meas[ok]
    if len(centres) < 4:
        return None, []
    ref = reference_ltass_db(centres)
    anchor = (centres >= 400.0) & (centres <= 800.0)
    if not anchor.any():
        anchor = np.ones(len(centres), dtype=bool)
    diff = (ref - ref[anchor].mean()) - (meas - meas[anchor].mean())
    kern = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
    kern /= kern.sum()
    diff_s = np.convolve(np.pad(diff, 3, mode="edge"), kern, mode="valid")
    corr = np.clip(diff_s, -max_db, max_db) * strength
    low = centres < 150.0
    corr[low] = np.minimum(corr[low], 0.0)

    nyq = sr / 2.0
    pts_f = np.concatenate([[0.0, f_lo / 2 ** (1.0 / 3)], centres, [min(f_hi * 2 ** (1.0 / 3), 0.999 * nyq), nyq]])
    pts_g = np.concatenate([[0.0, 0.0], corr, [0.0, 0.0]])
    order = np.argsort(pts_f, kind="stable")
    pts_f, pts_g = pts_f[order], pts_g[order]
    keep = np.concatenate([[True], np.diff(pts_f) > 0])
    pts_f, pts_g = pts_f[keep], pts_g[keep]
    pts_f[-1] = nyq
    numtaps = numtaps or (int(round(0.09 * sr)) | 1)
    taps = firwin2(numtaps, pts_f / nyq, 10 ** (pts_g / 20.0))
    bands = [{"centre_hz": round(float(c), 1), "measured_db": round(float(m), 2),
              "reference_db": round(float(r), 2), "correction_db": round(float(g), 2)}
             for c, m, r, g in zip(centres, meas - meas[anchor].mean(), ref - ref[anchor].mean(), corr)]
    return taps, bands


def apply_fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Zero-delay application of an odd-length linear-phase FIR."""
    return fftconvolve(x, taps, mode="same")
