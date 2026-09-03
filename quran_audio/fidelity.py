"""Fidelity self-check: did the restoration leave the voice alone?

Compares the input with the output of the restoration stages (before EQ
and level changes) and reports, per run, the numbers a reviewer would
otherwise have to measure by hand: how much of the voice band survived
in speech frames, how much the pauses dropped, whether onsets were
softened, whether the room's decay after phrases was cut, and how many
repairs touched speech. Bounds turn into warnings in the report, so the
promise "nothing of the voice is removed" is checked on every file.
"""
from __future__ import annotations

import numpy as np

from .stft import STFT, frame_length_for

BAND_CENTRES = 1000.0 * 2.0 ** (np.arange(-9, 10) / 3.0)      # 125 Hz .. 8 kHz, 1/3 octave
VOICE_BAND = (300.0, 3000.0)

# bounds (dB) that raise a warning
MAX_VOICE_LOSS_DB = 1.0
MAX_ONSET_LOSS_DB = 2.0
MAX_TAIL_CUT_DB = 3.0
MAX_PROJECTION_LOSS_DB = 1.0

_EPS = 1e-20


def _db(v):
    return 10.0 * np.log10(np.maximum(v, _EPS))


def _frame_stats(x: np.ndarray, st: STFT, fr: np.ndarray, band_masks: list[np.ndarray],
                 inb: np.ndarray, mid: np.ndarray, wide: np.ndarray):
    n_frames = st.n_frames(len(x))
    e_inb, e_mid, e_wide = np.empty(n_frames), np.empty(n_frames), np.empty(n_frames)
    bands = np.empty((n_frames, len(band_masks)), dtype=np.float32)
    for m0, power in st.power_blocks(x):
        m1 = m0 + power.shape[0]
        e_inb[m0:m1] = power[:, inb].mean(axis=1)
        e_mid[m0:m1] = power[:, mid].mean(axis=1)
        e_wide[m0:m1] = power[:, wide].mean(axis=1)
        for k, bm in enumerate(band_masks):
            bands[m0:m1, k] = power[:, bm].sum(axis=1)
    return e_inb, e_mid, e_wide, bands


def measure(x_in: np.ndarray, pre_eq: np.ndarray, sr: int, declick_report: dict | None = None) -> dict:
    """Fidelity of `pre_eq` (restoration output before EQ/level stages)
    against `x_in` (the same channel after DC removal). Both 1-D, equal length."""
    x_in = np.asarray(x_in, dtype=np.float64)
    pre_eq = np.asarray(pre_eq, dtype=np.float64)
    st = STFT(frame_length_for(sr))
    fr = st.freqs(sr)
    fps = sr / st.hop
    centres = [c for c in BAND_CENTRES if c * 2 ** (1 / 6) < 0.5 * sr]
    band_masks = [(fr >= c / 2 ** (1 / 6)) & (fr < c * 2 ** (1 / 6)) for c in centres]
    inb = (fr >= 100.0) & (fr <= min(4000.0, 0.45 * sr))
    mid = (fr >= 200.0) & (fr < min(4000.0, 0.45 * sr))
    wide = (fr >= 200.0) & (fr < min(6000.0, 0.45 * sr))
    a_inb, a_mid, a_wide, a_bands = _frame_stats(x_in, st, fr, band_masks, inb, mid, wide)
    _, b_mid, b_wide, b_bands = _frame_stats(pre_eq, st, fr, band_masks, inb, mid, wide)

    e_db = _db(a_inb)
    seed = float(np.percentile(e_db, 5))
    near = e_db[e_db < seed + 6.0]
    floor = float(np.median(near)) if near.size else seed
    speech = e_db > floor + 15.0
    pause = e_db < floor + 4.5
    sp10 = e_db > floor + 10.0

    out: dict = {"bands_hz": [round(float(c), 1) for c in centres]}
    if speech.sum() >= 10:
        ret = _db(b_bands[speech].sum(axis=0)) - _db(a_bands[speech].sum(axis=0))
        out["speech_retention_db"] = [round(float(v), 2) for v in ret]
        vb = np.array([(VOICE_BAND[0] <= c <= VOICE_BAND[1]) for c in centres])
        out["voice_band_retention_db"] = round(float(np.mean(ret[vb])), 2) if vb.any() else None
    else:
        out["speech_retention_db"] = None
        out["voice_band_retention_db"] = None
    if pause.sum() >= 10:
        out["pause_reduction_db"] = [round(float(v), 2) for v in _db(b_bands[pause].sum(axis=0)) - _db(a_bands[pause].sum(axis=0))]
    else:
        out["pause_reduction_db"] = None
    denom = float(np.dot(x_in, x_in))
    out["projection_db"] = round(float(20 * np.log10(max(np.dot(pre_eq, x_in) / denom, 1e-9))), 2) if denom > 0 else None

    # onsets: >= 0.5 s quiet before, >= 0.1 s speech after; first frame
    q, a = int(0.5 * fps), max(1, int(0.1 * fps))
    onsets = [i for i in range(q, len(sp10) - a) if sp10[i] and not sp10[i - 1] and not sp10[i - q:i].any() and sp10[i:i + a].mean() > 0.9]
    out["onsets"] = len(onsets)
    out["onset_retention_db"] = round(float(np.median([_db(b_wide[i]) - _db(a_wide[i]) for i in onsets])), 2) if len(onsets) >= 3 else None

    # phrase offsets: >= 1 s speech before, >= 0.6 s without after; tail at
    # 300 ms, counted only where the input still has a decay there (at least
    # 3 dB above the pause floor), otherwise cutting it is noise reduction
    pre_n, post_n, k = int(1.0 * fps), int(0.6 * fps), int(0.3 * fps)
    mid_floor = float(np.median(a_mid[pause])) if pause.sum() >= 10 else 0.0
    offsets = [i for i in range(pre_n, len(sp10) - post_n) if sp10[i - 1] and not sp10[i]
               and sp10[i - pre_n:i].mean() > 0.9 and not sp10[i:i + post_n].any()
               and a_mid[i + k] > 2.0 * mid_floor]
    out["offsets"] = len(offsets)
    if len(offsets) >= 3:
        tail_in = np.median([_db(a_mid[i + k]) - _db(a_mid[i - 1]) for i in offsets])
        tail_out = np.median([_db(b_mid[i + k]) - _db(a_mid[i - 1]) for i in offsets])
        out["tail_300ms_db"] = {"input": round(float(tail_in), 2), "output": round(float(tail_out), 2),
                                "extra_cut": round(float(tail_out - tail_in), 2)}
    else:
        out["tail_300ms_db"] = None
    out["in_speech_repairs"] = int(declick_report.get("in_speech_repairs", 0)) if declick_report else 0

    warnings: list[str] = []
    if out["voice_band_retention_db"] is not None and out["voice_band_retention_db"] < -MAX_VOICE_LOSS_DB:
        warnings.append(f"voice band (300-3000 Hz) lost {-out['voice_band_retention_db']:.1f} dB in speech frames")
    if out["projection_db"] is not None and out["projection_db"] < -MAX_PROJECTION_LOSS_DB:
        warnings.append(f"overall voice level dropped {-out['projection_db']:.1f} dB")
    if out["onset_retention_db"] is not None and out["onset_retention_db"] < -MAX_ONSET_LOSS_DB:
        warnings.append(f"onsets softened by {-out['onset_retention_db']:.1f} dB in their first frame")
    if out["tail_300ms_db"] is not None and out["tail_300ms_db"]["extra_cut"] < -MAX_TAIL_CUT_DB:
        warnings.append(f"room decay cut {-out['tail_300ms_db']['extra_cut']:.1f} dB more than the input 300 ms after phrases")
    out["warnings"] = warnings
    return out


def summary(f: dict) -> str:
    parts = []
    if f.get("voice_band_retention_db") is not None:
        parts.append(f"voice band {f['voice_band_retention_db']:+.2f} dB")
    if f.get("onset_retention_db") is not None:
        parts.append(f"onsets {f['onset_retention_db']:+.1f} dB")
    if f.get("tail_300ms_db"):
        parts.append(f"tail {f['tail_300ms_db']['extra_cut']:+.1f} dB")
    parts.append(f"in-speech repairs {f.get('in_speech_repairs', 0)}")
    return ", ".join(parts) + (f" | WARNINGS: {'; '.join(f['warnings'])}" if f.get("warnings") else "")
