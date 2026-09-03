"""Stereo transfers: measurement and folding strategies."""
import numpy as np

from conftest import at_snr, make_voice, white
from quran_audio import stereo


def _noise_of(fold, voice):
    a = np.dot(fold, voice) / np.dot(voice, voice)
    return 10 * np.log10(np.mean((fold - a * voice) ** 2))


def test_dual_mono_sums_coherently(voice, sr):
    n = len(voice)
    left = voice + at_snr(voice, white(n, 1), 25)
    right = voice + at_snr(voice, white(n, 2), 25)
    fold, rep = stereo.fold(np.stack([left, right], 1), sr)
    assert rep.strategy == "coherent-sum" and rep.coherence > 0.9 and rep.lag_samples == 0
    # independent noise averages down by ~3 dB
    assert _noise_of(fold, voice) < _noise_of(left, voice) - 2.0


def test_dead_channel_is_dropped(voice, sr):
    n = len(voice)
    left = voice + at_snr(voice, white(n, 1), 25)
    right = 0.002 * white(n, 4)
    fold, rep = stereo.fold(np.stack([left, right], 1), sr)
    assert rep.strategy == "best" and "dead" in rep.reason and np.allclose(fold, left)


def test_inverted_polarity_is_corrected(voice, sr):
    n = len(voice)
    left = voice + at_snr(voice, white(n, 1), 25)
    right = -(voice + at_snr(voice, white(n, 2), 25))
    fold, rep = stereo.fold(np.stack([left, right], 1), sr)
    assert rep.strategy == "coherent-sum" and rep.correlation < 0
    assert any("polarity" in note for note in rep.notes)
    assert np.corrcoef(fold, voice)[0, 1] > 0.99


def test_delayed_channel_is_aligned(voice, sr):
    """A 20-sample inter-channel delay comb-filters a naive average; the
    coherent sum realigns first and keeps the top of the band."""
    from scipy.signal import butter, sosfiltfilt
    n = len(voice)
    left = voice + at_snr(voice, white(n, 1), 30)
    right = np.roll(voice, 20) + at_snr(voice, white(n, 2), 30)
    fold, rep = stereo.fold(np.stack([left, right], 1), sr)
    assert rep.strategy == "coherent-sum" and rep.lag_samples == 20
    assert any("aligned" in note for note in rep.notes)
    naive = 0.5 * (left + right)
    # the naive average carries a combed copy of the voice as error; the
    # aligned sum leaves only noise
    assert _noise_of(naive, voice) - _noise_of(fold, voice) > 6.0
    sos = butter(4, [1000, 3000], btype="band", fs=sr, output="sos")
    assert np.std(sosfiltfilt(sos, fold)) > np.std(sosfiltfilt(sos, naive))


def test_different_content_picks_the_better_channel(voice, sr):
    """Two channels that are not the same signal must not be averaged."""
    n = len(voice)
    other = make_voice(sr, len(voice) / sr, f0=190.0, seed=5)[:n]
    left = voice + at_snr(voice, white(n, 1), 20)
    right = other + at_snr(other, white(n, 2), 30)
    fold, rep = stereo.fold(np.stack([left, right], 1), sr)
    assert rep.strategy == "best" and rep.coherence < stereo.COHERENT_THRESHOLD
    assert np.allclose(fold, right)          # the cleaner channel
    assert rep.to_dict()["strategy"] == "best"


def test_explicit_strategies(voice, sr):
    n = len(voice)
    pair = np.stack([voice, 0.5 * voice + at_snr(voice, white(n, 2), 30)], 1)
    for strategy in ("mix", "left", "right", "best", "coherent-sum"):
        fold, rep = stereo.fold(pair, sr, strategy)
        assert len(fold) == n and rep.strategy == strategy
