import numpy as np

from quran_audio import dynamics


def test_loudness_and_true_peak_of_reference_sine():
    sr = 44100
    t = np.arange(sr * 5) / sr
    sine = 10 ** (-20 / 20) * np.sin(2 * np.pi * 1000 * t)
    assert abs(dynamics.integrated_loudness(sine, sr) + 23.0) < 0.1
    assert abs(dynamics.true_peak_dbtp(sine) + 20.0) < 0.05
    assert dynamics.integrated_loudness(np.zeros(sr), sr) == float("-inf")
    stereo = np.stack([sine, sine], 1)
    assert abs(dynamics.integrated_loudness(stereo, sr) + 20.0) < 0.1


def test_limiter_holds_ceiling(voice, sr):
    hot = voice * 4.0
    hot[sr] = 1.5
    hot[2 * sr:2 * sr + 3] = [-1.2, 1.4, -1.1]
    y, info = dynamics.limiter(hot, sr, ceiling_db=-1.0)
    assert info["applied"] and len(y) == len(hot)
    assert dynamics.true_peak_dbtp(y) <= -1.0 + 0.05
    quiet, info2 = dynamics.limiter(voice * 0.5, sr)
    assert not info2["applied"] and quiet is not None


def test_normalize_hits_target(voice, sr):
    y, info = dynamics.normalize_loudness(voice, sr, -18.0, -1.0)
    assert abs(info["loudness_after_lufs"] + 18.0) < 0.5
    assert info["true_peak_after_dbtp"] <= -1.0 + 0.05


def test_leveler_reduces_spread(voice, sr):
    drift = voice * 10 ** (np.linspace(-6, 6, len(voice)) / 20)
    y, info = dynamics.leveler(drift, sr)
    assert info["applied"] and len(y) == len(drift)

    def spread(x):
        lv = dynamics.frame_rms_db(x, sr)[1]
        lv = lv[lv > -40]
        return np.diff(np.percentile(lv, [10, 90]))[0]

    assert spread(y) < spread(drift) - 2
    silent, info2 = dynamics.leveler(np.zeros(sr), sr)
    assert not info2["applied"]
