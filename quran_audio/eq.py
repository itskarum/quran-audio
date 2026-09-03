"""Voice-band conditioning: rumble high-pass, bandwidth-aware low-pass and a
conservative tonal-balance correction toward the long-term average speech
spectrum. Every filter is zero-phase, so the output stays sample-aligned.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, firwin2, oaconvolve, sosfiltfilt


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


# Long-term spectrum of a clean studio recitation (male reciter, 135 s,
# 44.1 kHz), 1/3-octave band levels relative to the 400-800 Hz mean,
# noise-subtracted and smoothed over three bands. Measured for this
# project; it is what tajweed recitation looks like as opposed to
# conversational speech: 7-12 dB fuller below 315 Hz, 10-14 dB darker
# above 1.5 kHz. Replace it with `--tonal-reference FILE` (a good recording
# of the same reciter is the best possible target).
RECITATION_LTAS_DB: dict[float, float] = {
    125.0: 3.2, 157.5: 4.0, 198.4: 3.4, 250.0: 4.9, 315.0: 5.7, 396.9: 3.0, 500.0: 0.1, 630.0: -3.0,
    793.7: -7.4, 1000.0: -10.1, 1259.9: -9.9, 1587.4: -11.4, 2000.0: -15.0, 2519.8: -16.8,
    3174.8: -20.1, 4000.0: -26.0, 5039.7: -30.3, 6349.6: -32.4, 8000.0: -33.6,
}


def reference_recitation_db(freqs: np.ndarray) -> np.ndarray:
    """The recitation reference interpolated in log-frequency; held flat
    beyond the measured range."""
    f = np.array(sorted(RECITATION_LTAS_DB))
    v = np.array([RECITATION_LTAS_DB[k] for k in f])
    return np.interp(np.log(np.maximum(np.asarray(freqs, dtype=np.float64), 1.0)), np.log(f), v)


def reference_from_file(path, sr_hint: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(centres, levels_db) of the long-term speech spectrum of a reference
    recording, for use as the tonal-balance target."""
    from .audio_io import load, to_mono
    from .analysis import speech_spectrum
    a = load(path)
    mono = to_mono(a.samples, sr=a.sample_rate)[0]
    freqs, sp, nz = speech_spectrum(mono - mono.mean(), a.sample_rate)
    clean = np.maximum(sp - nz, nz)
    centres = third_octave_centres(100.0, min(10000.0, 0.45 * a.sample_rate))
    return centres, band_levels_db(clean, freqs, centres)


def resolve_reference(reference, centres: np.ndarray) -> np.ndarray:
    """Reference levels at `centres` from a name ("recitation", "speech"),
    a (centres, levels) pair, or a file path."""
    if reference is None or reference == "recitation":
        return reference_recitation_db(centres)
    if reference == "speech":
        return reference_ltass_db(centres)
    if isinstance(reference, (tuple, list)) and len(reference) == 2:
        rc, rv = np.asarray(reference[0], dtype=np.float64), np.asarray(reference[1], dtype=np.float64)
        ok = np.isfinite(rv)
        return np.interp(np.log(centres), np.log(rc[ok]), rv[ok])
    return resolve_reference(reference_from_file(reference), centres)


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
                         numtaps: int | None = None, reference="recitation") -> tuple[np.ndarray | None, list[dict]]:
    """Linear-phase FIR that nudges the measured speech spectrum toward the
    reference shape. Corrections are smoothed over one octave (so formants
    are never carved), clamped to +-max_db, scaled by `strength`, and:
    never boost below 150 Hz, at most +2 dB below 300 Hz (hum and rumble
    territory), at most +1.5 dB above 5 kHz (where an old transfer holds
    little but hiss), nothing where the voice itself is more than 20 dB
    below its 500 Hz region (nothing there to restore), and zero outside
    the usable band."""
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
    ref = resolve_reference(reference, centres)
    anchor = (centres >= 400.0) & (centres <= 800.0)
    if not anchor.any():
        anchor = np.ones(len(centres), dtype=bool)
    meas_rel = meas - meas[anchor].mean()
    diff = (ref - ref[anchor].mean()) - meas_rel
    kern = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
    kern /= kern.sum()
    diff_s = np.convolve(np.pad(diff, 3, mode="edge"), kern, mode="valid")
    corr = np.clip(diff_s, -max_db, max_db) * strength
    corr[centres < 150.0] = np.minimum(corr[centres < 150.0], 0.0)
    corr[centres < 300.0] = np.minimum(corr[centres < 300.0], 2.0)
    corr[centres > 5000.0] = np.minimum(corr[centres > 5000.0], 1.5)
    corr[meas_rel < -20.0] = np.minimum(corr[meas_rel < -20.0], 0.0)

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
             for c, m, r, g in zip(centres, meas_rel, ref - ref[anchor].mean(), corr)]
    return taps, bands


def apply_fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Zero-delay application of an odd-length linear-phase FIR. Overlap-add
    (block FFTs), so an hour-long file never needs a full-length FFT."""
    return oaconvolve(x, taps, mode="same")
