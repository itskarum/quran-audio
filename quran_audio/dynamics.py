"""Level management: a slow speech leveler, ITU-R BS.1770-4 integrated
loudness, true-peak measurement and a look-ahead true-peak limiter.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import minimum_filter1d
from scipy.signal import lfilter, resample_poly

from .audio_io import _resample_filter, resample

# BS.1770-4 K-weighting biquads, defined at 48 kHz
_PRE_B = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_PRE_A = [1.0, -1.69065929318241, 0.73248077421585]
_RLB_B = [1.0, -2.0, 1.0]
_RLB_A = [1.0, -1.99004745483398, 0.99007225036621]
_OS_FILTER = _resample_filter(4, 1)


def _as_2d(x: np.ndarray) -> np.ndarray:
    return x[:, None] if x.ndim == 1 else x


# ----- leveler ------------------------------------------------------------

def frame_rms_db(x: np.ndarray, sr: int, win_s: float = 0.05, hop_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    if len(x) < win:
        return np.array([len(x) / 2.0]), np.array([10.0 * np.log10(np.mean(x * x) + 1e-20)])
    frames = sliding_window_view(x, win)[::hop]
    levels = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-20)
    centres = np.arange(len(levels)) * hop + win / 2.0
    return centres, levels


def leveler(x: np.ndarray, sr: int, range_db: float = 6.0, ratio: float = 0.5,
            attack_s: float = 0.1, release_s: float = 1.5,
            speech_margin_db: float = 10.0) -> tuple[np.ndarray, dict]:
    """Gently pull speech frames toward their median level (a 2:1 leveler at
    ratio 0.5), bounded to +-range_db. The gain freezes during pauses so
    residual noise is never pumped up between phrases."""
    centres, levels = frame_rms_db(x, sr)
    seed = float(np.percentile(levels, 5))
    near = levels[levels < seed + 6.0]
    floor = float(np.median(near)) if near.size else seed
    speech = levels > floor + speech_margin_db
    if speech.sum() < 10:
        return x, {"applied": False, "reason": "too little speech-level material"}
    target = float(np.median(levels[speech]))
    desired = np.clip((target - levels) * ratio, -range_db, range_db)
    idx = np.where(speech, np.arange(len(levels)), 0)
    np.maximum.accumulate(idx, out=idx)
    held = np.where(speech[idx], desired[idx], 0.0)
    hop_s = 0.01
    c_att = 1.0 - np.exp(-hop_s / attack_s)
    c_rel = 1.0 - np.exp(-hop_s / release_s)
    g = np.empty_like(held)
    prev = 0.0
    for i, d in enumerate(held):
        prev += (d - prev) * (c_att if d < prev else c_rel)
        g[i] = prev
    gain = np.interp(np.arange(len(x)), centres, 10 ** (g / 20.0))
    return x * gain, {"applied": True, "target_db": round(target, 2),
                      "gain_min_db": round(float(g.min()), 2), "gain_max_db": round(float(g.max()), 2)}


# ----- loudness -----------------------------------------------------------

def k_weight_48k(x: np.ndarray) -> np.ndarray:
    return lfilter(_RLB_B, _RLB_A, lfilter(_PRE_B, _PRE_A, x))


def integrated_loudness(x: np.ndarray, sr: int) -> float:
    """BS.1770-4 integrated loudness in LUFS (gated). -inf for silence."""
    x2 = _as_2d(np.asarray(x, dtype=np.float64))
    chans = [k_weight_48k(resample(x2[:, c], sr, 48000)) for c in range(x2.shape[1])]
    block, hop = 19200, 4800
    n = len(chans[0])
    if n < block:
        chans = [np.concatenate([c, np.zeros(block - n)]) for c in chans]
    ms = np.zeros(len(sliding_window_view(chans[0], block)[::hop]))
    for c in chans:
        frames = sliding_window_view(c, block)[::hop]
        ms += np.mean(frames * frames, axis=1)
    lk = -0.691 + 10.0 * np.log10(ms + 1e-20)
    abs_gate = lk > -70.0
    if not abs_gate.any():
        return float("-inf")
    rel = -0.691 + 10.0 * np.log10(np.mean(ms[abs_gate])) - 10.0
    gate = abs_gate & (lk > rel)
    if not gate.any():
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(np.mean(ms[gate])))


def _true_peak_envelope(x2: np.ndarray) -> np.ndarray:
    """Per-sample max of the 4x oversampled absolute signal, across channels."""
    n = x2.shape[0]
    tp = np.zeros(n)
    for c in range(x2.shape[1]):
        os = resample_poly(x2[:, c], 4, 1, window=_OS_FILTER)
        tp = np.maximum(tp, np.abs(os[:4 * n]).reshape(n, 4).max(axis=1))
    return tp


def true_peak_dbtp(x: np.ndarray, sr: int | None = None) -> float:
    tp = _true_peak_envelope(_as_2d(np.asarray(x, dtype=np.float64)))
    return float(20.0 * np.log10(max(float(tp.max()), 1e-12)))


# ----- limiter ------------------------------------------------------------

def limiter(x: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead_ms: float = 5.0,
            release_ms: float = 80.0) -> tuple[np.ndarray, dict]:
    """Look-ahead true-peak limiter. The gain ramps down over `lookahead_ms`
    ahead of every peak (so attacks are never clipped or distorted) and
    releases exponentially. Channels are gain-linked."""
    x2 = _as_2d(np.asarray(x, dtype=np.float64))
    n = x2.shape[0]
    ceiling = 10 ** (ceiling_db / 20.0)
    tp = _true_peak_envelope(x2)
    g_req = np.minimum(1.0, ceiling / np.maximum(tp, 1e-12))
    if g_req.min() >= 1.0:
        return x, {"applied": False, "max_reduction_db": 0.0}
    la = max(1, int(lookahead_ms * 1e-3 * sr))
    # look-ahead minimum: h[i] = min(g_req[i : i + 2 la])
    padded = np.concatenate([g_req, np.ones(la)])
    h = minimum_filter1d(padded, size=2 * la, mode="constant", cval=1.0)[la:la + n]
    # exponential release at a 1 ms control rate
    ctrl = max(1, int(0.001 * sr))
    m = -(-n // ctrl)
    hp = np.concatenate([h, np.ones(m * ctrl - n)]).reshape(m, ctrl).min(axis=1)
    r = 1.0 - np.exp(-ctrl / (release_ms * 1e-3 * sr))
    gc = np.empty(m)
    prev = 1.0
    for i in range(m):
        prev = min(hp[i], prev + (1.0 - prev) * r)
        gc[i] = prev
    g = np.interp(np.arange(n), np.arange(m) * ctrl + ctrl / 2.0, gc)
    g = np.minimum(g, h)
    # causal moving average over la samples: the attack ramp
    v = np.concatenate([np.ones(la), g])
    cs = np.concatenate([[0.0], np.cumsum(v)])
    g_final = (cs[la + 1:la + 1 + n] - cs[1:1 + n]) / la
    y = x2 * g_final[:, None]
    over = int(np.count_nonzero(np.abs(y) > ceiling * 1.001))
    if x.ndim == 1:
        y = y[:, 0]
    return y, {"applied": True, "max_reduction_db": round(float(20 * np.log10(g_final.min())), 2),
               "samples_over_ceiling": over}


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -18.0, ceiling_db: float = -1.0,
                       max_gain_db: float = 30.0) -> tuple[np.ndarray, dict]:
    before = integrated_loudness(x, sr)
    gain_db = 0.0 if not np.isfinite(before) else float(np.clip(target_lufs - before, -max_gain_db, max_gain_db))
    y = x * 10 ** (gain_db / 20.0)
    y, lim = limiter(y, sr, ceiling_db)
    after = integrated_loudness(y, sr)
    return y, {"loudness_before_lufs": round(before, 2) if np.isfinite(before) else None,
               "gain_db": round(gain_db, 2),
               "loudness_after_lufs": round(after, 2) if np.isfinite(after) else None,
               "true_peak_after_dbtp": round(true_peak_dbtp(y), 2), "limiter": lim}
