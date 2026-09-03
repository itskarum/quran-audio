"""Stereo transfers: measure how the two channels relate, then fold or
link them accordingly.

Old recordings arrive as stereo files for many reasons: dual-mono
transfers, two microphones in a hall, pseudo-stereo from an upload, or
two tape tracks with their own noise. Averaging only helps when the
channels carry the same signal with independent noise. The measurements
here (level difference, delay, polarity, voice-band coherence, per-channel
noise floor) decide which strategy applies, and the decision is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, coherence, sosfiltfilt

from .dynamics import frame_rms_db

VOICE_BAND = (150.0, 3000.0)
COHERENT_THRESHOLD = 0.7      # below this the channels are not the same signal
DEAD_CHANNEL_DB = 20.0        # a channel this much quieter than the other is dead
MAX_LAG_MS = 5.0


@dataclass
class StereoReport:
    level_db: list[float]
    level_diff_db: float
    correlation: float            # at the best lag, sign kept
    lag_samples: int              # right relative to left (positive: right is late)
    coherence: float              # mean magnitude-squared coherence over the voice band
    snr_est_db: list[float]       # speech level over noise floor, per channel
    strategy: str = ""
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level_db": [round(v, 2) for v in self.level_db],
            "level_diff_db": round(self.level_diff_db, 2),
            "correlation": round(self.correlation, 3),
            "lag_samples": int(self.lag_samples),
            "coherence_voice_band": round(self.coherence, 3),
            "snr_est_db": [round(v, 1) for v in self.snr_est_db],
            "strategy": self.strategy,
            "reason": self.reason,
            "notes": list(self.notes),
        }


def _snr_estimate_db(x: np.ndarray, sr: int) -> float:
    levels = frame_rms_db(x, sr)[1]
    seed = float(np.percentile(levels, 5))
    near = levels[levels < seed + 6.0]
    floor = float(np.median(near)) if near.size else seed
    speech = levels[levels > floor + 10.0]
    return float(np.median(speech) - floor) if speech.size else 0.0


def _excerpt(samples: np.ndarray, sr: int, seconds: float = 120.0) -> np.ndarray:
    n = samples.shape[0]
    take = min(n, int(seconds * sr))
    start = (n - take) // 2
    return samples[start:start + take]


def measure(samples: np.ndarray, sr: int) -> StereoReport:
    """Measure a (n, 2) signal. Levels and noise floors use the whole file;
    delay, polarity and coherence use up to two minutes from the middle."""
    x = np.asarray(samples, dtype=np.float64)
    left, right = x[:, 0] - x[:, 0].mean(), x[:, 1] - x[:, 1].mean()
    level = [20 * np.log10(np.std(c) + 1e-12) for c in (left, right)]
    ex = _excerpt(np.stack([left, right], axis=1), sr)
    sos = butter(4, [100.0, min(4000.0, 0.45 * sr)], btype="band", fs=sr, output="sos")
    a, b = sosfiltfilt(sos, ex[:, 0]), sosfiltfilt(sos, ex[:, 1])
    k = max(1, int(MAX_LAG_MS * 1e-3 * sr))
    lags = np.arange(-k, k + 1)
    norm = np.sqrt(np.dot(a[k:-k], a[k:-k]) * np.dot(b[k:-k], b[k:-k])) + 1e-20
    cc = np.array([np.dot(a[k:-k], b[k + lag:len(b) - k + lag]) for lag in lags]) / norm
    best = int(np.argmax(np.abs(cc)))
    lag, corr = int(lags[best]), float(cc[best])
    f, coh = coherence(ex[:, 0], ex[:, 1], fs=sr, nperseg=2048)
    sel = (f >= VOICE_BAND[0]) & (f <= VOICE_BAND[1])
    coh_voice = float(np.mean(coh[sel])) if sel.any() else 0.0
    snr = [_snr_estimate_db(left, sr), _snr_estimate_db(right, sr)]
    return StereoReport(level, level[0] - level[1], corr, lag, coh_voice, snr)


def fold(samples: np.ndarray, sr: int, strategy: str = "auto",
         report: StereoReport | None = None) -> tuple[np.ndarray, StereoReport]:
    """Fold (n, 2) to mono. Strategies: auto, best, coherent-sum, mix, left, right."""
    x = np.asarray(samples, dtype=np.float64)
    rep = report or measure(x, sr)
    left, right = x[:, 0], x[:, 1]
    if strategy == "auto":
        if abs(rep.level_diff_db) >= DEAD_CHANNEL_DB:
            strategy, rep.reason = "best", f"one channel is {abs(rep.level_diff_db):.0f} dB down: dead"
        elif rep.coherence >= COHERENT_THRESHOLD:
            strategy, rep.reason = "coherent-sum", f"voice-band coherence {rep.coherence:.2f}: same signal, sum coherently"
        else:
            strategy, rep.reason = "best", f"voice-band coherence {rep.coherence:.2f}: channels differ, averaging would cancel voice"
    rep.strategy = strategy
    if strategy == "left":
        return left.copy(), rep
    if strategy == "right":
        return right.copy(), rep
    if strategy == "mix":
        return 0.5 * (left + right), rep
    if strategy == "best":
        score = [rep.snr_est_db[0] + 0.1 * rep.level_db[0], rep.snr_est_db[1] + 0.1 * rep.level_db[1]]
        pick = int(np.argmax(score))
        rep.notes.append(f"kept {'left' if pick == 0 else 'right'} (SNR {rep.snr_est_db[pick]:.1f} dB vs {rep.snr_est_db[1 - pick]:.1f} dB)")
        return (left if pick == 0 else right).copy(), rep
    if strategy == "coherent-sum":
        r = right.copy()
        if rep.correlation < 0:
            r = -r
            rep.notes.append("right channel polarity inverted; corrected")
        if rep.lag_samples:
            r = np.roll(r, -rep.lag_samples)
            rep.notes.append(f"right channel aligned by {rep.lag_samples} samples")
        # level-match to the louder channel so the sum is not dominated by one side
        g = 10 ** ((rep.level_db[0] - rep.level_db[1]) / 20.0)
        if g < 1.0:
            left = left * (1.0 / g)
            rep.notes.append(f"left channel raised {-20 * np.log10(g):.1f} dB to match")
        else:
            r = r * g
            if g > 1.0001:
                rep.notes.append(f"right channel raised {20 * np.log10(g):.1f} dB to match")
        return 0.5 * (left + r), rep
    raise ValueError(f"unknown mono strategy {strategy!r}")
