"""Shared synthetic material: a recitation-like voice with breath pauses,
plus helpers to degrade it. Deterministic (seeded) so thresholds are stable."""
from __future__ import annotations

import numpy as np
import pytest

SR = 22050


def make_voice(sr: int = SR, dur: float = 12.0, f0: float = 140.0, seed: int = 0,
               pause_every: float = 5.0, pause_len: float = 0.8, vibrato: float = 0.02) -> np.ndarray:
    """Harmonic series with vibrato and slow drift, formant-like weighting,
    a syllable-rate amplitude envelope and silent breath pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    vib = 1 + vibrato * np.sin(2 * np.pi * 5.5 * t) + 0.05 * np.sin(2 * np.pi * 0.2 * t)
    phase = 2 * np.pi * np.cumsum(f0 * vib) / sr
    y = np.zeros_like(t)
    for k in range(1, 80):
        fk = k * f0
        if fk > 0.45 * sr:
            break
        form = (np.exp(-((fk - 600) / 400) ** 2) + 0.5 * np.exp(-((fk - 1800) / 500) ** 2)
                + 0.25 * np.exp(-((fk - 3000) / 600) ** 2) + 0.02)
        y += (form / k ** 0.5) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 2.3 * t) ** 2
    for start in np.arange(pause_every - 1, dur, pause_every):
        i0, i1 = int(start * sr), min(len(t), int((start + pause_len) * sr))
        ramp = max(1, int(0.03 * sr))
        env[i0:i1] = 0
        env[max(0, i0 - ramp):i0] *= np.linspace(1, 0, min(ramp, i0))
        env[i1:i1 + ramp] *= np.linspace(0, 1, len(env[i1:i1 + ramp]))
    y *= env
    return 0.3 * y / np.max(np.abs(y))


def white(n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so that clean + noise has the requested SNR."""
    return noise * np.sqrt(np.sum(clean ** 2) / np.sum(noise ** 2)) * 10 ** (-snr_db / 20)


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(ref ** 2) / (np.sum((est - ref) ** 2) + 1e-20)))


def si_sdr_db(ref: np.ndarray, est: np.ndarray) -> float:
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    s = a * ref
    return float(10 * np.log10(np.sum(s ** 2) / (np.sum((est - s) ** 2) + 1e-20)))


def pause_slice(sr: int = SR, which: int = 0) -> slice:
    start = 4.0 + 5.0 * which
    return slice(int((start + 0.05) * sr), int((start + 0.75) * sr))


@pytest.fixture(scope="session")
def voice() -> np.ndarray:
    return make_voice()


@pytest.fixture(scope="session")
def sr() -> int:
    return SR
