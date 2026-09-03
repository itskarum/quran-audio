import numpy as np

from conftest import at_snr, white
from quran_audio import declick


def _degrade(voice, sr, seed=2):
    rng = np.random.default_rng(seed)
    hiss = at_snr(voice, white(len(voice), 9), 45)
    base = voice + hiss
    pos = rng.integers(2000, len(voice) - 2000, 60)
    clicks = np.zeros_like(voice)
    for p in pos:
        clicks[p:p + rng.integers(1, 5)] = rng.uniform(0.15, 0.6) * rng.choice([-1, 1])
    plos_pos = [int(1.0 * sr), int(7.5 * sr)]
    plos = np.zeros_like(voice)
    for p in plos_pos:
        L = int(0.012 * sr)
        plos[p:p + L] = 0.25 * rng.standard_normal(L) * np.exp(-np.arange(L) / (0.003 * sr))
    return base + plos, base + plos + clicks, pos, plos_pos


def test_clicks_repaired_plosives_untouched(voice, sr):
    ref, degraded, pos, plos_pos = _degrade(voice, sr)
    y, rep = declick.declick(degraded, sr, threshold=6.0, max_click_ms=2.0)
    assert len(y) == len(degraded) and rep.clicks_repaired >= 40
    before = np.array([np.max(np.abs((degraded - ref)[p:p + 6])) for p in pos])
    after = np.array([np.max(np.abs((y - ref)[p:p + 6])) for p in pos])
    assert 20 * np.log10(np.median(before) / np.median(after)) > 15
    for p in plos_pos:
        L = int(0.012 * sr)
        assert np.array_equal(y[p - 100:p + L + 100], degraded[p - 100:p + L + 100])
    assert rep.skipped_long >= 1


def test_clean_input_untouched(voice, sr):
    x = voice + at_snr(voice, white(len(voice), 4), 40)
    y, rep = declick.declick(x, sr)
    assert rep.clicks_repaired == 0 and np.array_equal(y, x)


def test_protect_mask(voice, sr):
    _, degraded, pos, _ = _degrade(voice, sr)
    protect = np.ones(len(degraded), dtype=bool)
    y, rep = declick.declick(degraded, sr, protect=protect)
    assert rep.clicks_repaired == 0 and np.array_equal(y, degraded)


def test_declip_recovers_peaks(voice, sr):
    loud = voice * 2.2
    clipped = np.clip(loud, -0.5, 0.5)
    y, rep = declick.declip(clipped, sr)
    assert rep.clicks_repaired > 50 and len(y) == len(clipped)
    # never worse than leaving the peaks flat, and the peaks come back up
    assert np.std(y - loud) < np.std(clipped - loud)
    assert np.max(np.abs(y)) > 0.53
    assert np.all(np.abs(y)[np.abs(clipped) >= 0.5] >= 0.5)      # clipping-consistent


def test_lsar_fill_interpolates_sine():
    sr = 8000
    t = np.arange(400) / sr
    x = np.sin(2 * np.pi * 300 * t)
    a = declick.ar_coefficients(x, 8)
    unknown = np.zeros(400, dtype=bool)
    unknown[200:206] = True
    seg = x.copy()
    seg[unknown] = 0
    filled = declick.lsar_fill(seg, unknown, a)
    assert np.max(np.abs(filled[unknown] - x[unknown])) < 1e-3
    assert declick.ar_coefficients(np.zeros(400), 8) is None
