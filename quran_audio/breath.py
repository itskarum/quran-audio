"""Breath attenuation (opt-in).

Inhalations between phrases are part of the performance for some
listeners and a distraction for others. Mastering practice for recitation
is to attenuate them by a few decibels, never to remove them, so the
phrasing stays audible.

A breath here is a stretch of 120-900 ms that is noise-like (no periodicity
in the 70-400 Hz pitch range: the normalised autocorrelation of the 100 Hz
to 4 kHz band stays below 0.35, where a vowel measures above 0.5 even at
low level and even through the spectral holes of an MP3 transfer),
tilted upward (spectral centroid above 800 Hz, where a vowel or a decaying
room tail sits around 400-700 Hz), clearly above the noise floor but well
below the loud speech level, and not within 60 ms of a voiced frame. The
last rule keeps phrase-final fricatives and plosive releases, which are
also noise-like, out of the detector's reach: they abut the vowel that
precedes them. A breath that runs straight into the next phrase is left
alone for the same reason. Detected breaths are attenuated with 30 ms
raised-cosine edges.
"""
from __future__ import annotations

import numpy as np

from .stft import STFT, frame_length_for

MAX_ATTENUATION_DB = 12.0
MIN_SECONDS = 0.12
MAX_SECONDS = 0.9
GUARD_SECONDS = 0.06
MAX_HARMONICITY = 0.35
MIN_CENTROID_HZ = 800.0
MIN_ABOVE_FLOOR_DB = 8.0       # a breath must stand clearly out of the noise floor
MIN_BELOW_SPEECH_DB = 6.0      # and stay well below the loud speech level
PITCH_LAG_S = (0.0025, 0.014)  # 70-400 Hz


def frame_features(x: np.ndarray, sr: int) -> dict:
    """Per-frame level above the floor (dB), spectral centroid (Hz) and
    harmonicity (peak normalised autocorrelation in the pitch range) of
    the 100 Hz-4 kHz band."""
    x = np.asarray(x, dtype=np.float64)
    st = STFT(frame_length_for(sr))
    fr = st.freqs(sr)
    top = min(4000.0, 0.45 * sr)
    band = (fr >= 100.0) & (fr <= top)
    n_frames = st.n_frames(len(x))
    energy = np.zeros(n_frames)
    centroid = np.zeros(n_frames)
    harmonicity = np.zeros(n_frames)
    if n_frames < 4 or band.sum() < 4:
        return {"st": st, "rel": energy, "centroid": centroid, "harmonicity": harmonicity, "speech_db": 0.0}
    w = st.window
    w_acf = np.correlate(w, w, "full")[len(w) - 1:]
    lag0, lag1 = int(PITCH_LAG_S[0] * sr), min(int(PITCH_LAG_S[1] * sr), st.n_fft // 2)
    lag_norm = w_acf[0] / np.maximum(w_acf[lag0:lag1], 1e-9)
    for m0, power in st.power_blocks(x):
        m1 = m0 + power.shape[0]
        pb = power[:, band]
        energy[m0:m1] = pb.mean(axis=1)
        centroid[m0:m1] = (pb * fr[band]).sum(axis=1) / np.maximum(pb.sum(axis=1), 1e-30)
        acf = np.fft.irfft(np.where(band, power, 0.0), n=st.n_fft, axis=1)
        r = acf[:, lag0:lag1] / np.maximum(acf[:, :1], 1e-30) * lag_norm
        harmonicity[m0:m1] = r.max(axis=1)
    e_db = 10 * np.log10(np.maximum(energy, 1e-30))
    seed = float(np.percentile(e_db, 5))
    near = e_db[e_db < seed + 6.0]
    floor = float(np.median(near)) if near.size else seed
    rel = e_db - floor
    return {"st": st, "rel": rel, "centroid": centroid, "harmonicity": harmonicity,
            "speech_db": float(np.percentile(rel, 90))}


def find_breaths(x: np.ndarray, sr: int, details: list | None = None) -> list[tuple[int, int]]:
    """Sample ranges of breaths. `details`, if given, collects every
    candidate run with the reason it was kept or dropped."""
    f = frame_features(x, sr)
    st, rel, centroid, harmonicity = f["st"], f["rel"], f["centroid"], f["harmonicity"]
    n_frames = len(rel)
    if n_frames < 4 or f["speech_db"] < MIN_ABOVE_FLOOR_DB + MIN_BELOW_SPEECH_DB:
        return []
    ceiling = f["speech_db"] - MIN_BELOW_SPEECH_DB
    breathlike = (centroid >= MIN_CENTROID_HZ) & (harmonicity < MAX_HARMONICITY)
    voiced = (rel > 10.0) & (harmonicity >= MAX_HARMONICITY)
    candidate = (rel > 3.0) & breathlike
    fps = sr / st.hop
    lo, hi = int(round(MIN_SECONDS * fps)), int(round(MAX_SECONDS * fps))
    guard = max(1, int(round(GUARD_SECONDS * fps)))
    centre = st.n_fft // 2 - (st.n_fft - st.hop)
    out = []
    i = 0
    while i < n_frames:
        if not candidate[i]:
            i += 1
            continue
        j = i
        while j < n_frames and candidate[j]:
            j += 1
        peak = float(rel[i:j].max())
        a = max(0, i * st.hop + centre)
        b = min(len(x), j * st.hop + centre)
        if not lo <= j - i <= hi:
            reason = "duration"
        elif peak < MIN_ABOVE_FLOOR_DB:
            reason = "too quiet"
        elif peak > ceiling:
            reason = "too loud"
        elif voiced[max(0, i - guard):i].any() or voiced[j:j + guard].any():
            reason = "touches voice"
        else:
            reason = "kept"
            if b > a:
                out.append((int(a), int(b)))
        if details is not None:
            details.append({"start_s": round(a / sr, 3), "end_s": round(b / sr, 3), "peak_db": round(peak, 1),
                            "frames": int(j - i), "reason": reason})
        i = j
    return out


def breath_gain(x: np.ndarray, sr: int, db: float = -6.0) -> tuple[np.ndarray | None, dict]:
    """Per-sample gain that attenuates the detected breaths by `db` (capped
    at -12 dB; never removed), or None when there is nothing to do."""
    db = -min(abs(float(db)), MAX_ATTENUATION_DB)
    ranges = find_breaths(x, sr)
    info = {"count": len(ranges), "attenuation_db": db,
            "total_s": round(sum(b - a for a, b in ranges) / sr, 2),
            "ranges_s": [(round(a / sr, 3), round(b / sr, 3)) for a, b in ranges]}
    if not ranges or db >= 0:
        return None, info
    g = 10 ** (db / 20.0)
    edge = max(1, int(0.03 * sr))
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
    gain = np.ones(len(x))
    for a, b in ranges:
        seg = np.full(b - a, g)
        n_edge = min(edge, (b - a) // 2)
        if n_edge:
            seg[:n_edge] = 1.0 - (1.0 - g) * ramp[:n_edge]
            seg[-n_edge:] = 1.0 - (1.0 - g) * ramp[:n_edge][::-1]
        gain[a:b] = np.minimum(gain[a:b], seg)
    return gain, info


def attenuate_breaths(x: np.ndarray, sr: int, db: float = -6.0) -> tuple[np.ndarray, dict]:
    """Attenuate detected breaths by `db` (capped at -12 dB; never removed)."""
    x = np.asarray(x, dtype=np.float64)
    gain, info = breath_gain(x, sr, db)
    if gain is None:
        return x, info
    return x * gain, info
