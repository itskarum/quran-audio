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


def test_leveler_is_phrase_level():
    """The leveler follows phrases (seconds), never syllables (100 ms), and
    stays inside its +-3 dB range."""
    sr = 22050
    t = np.arange(int(8 * sr)) / sr
    carrier = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2        # ~9 dB swing at 4 Hz
    level = np.where(t < 4.0, 1.0, 10 ** (-10 / 20))                   # a 10 dB phrase step at 4 s
    for a in (1.8, 3.8, 5.8):                                          # breath pauses, so a real floor exists
        level[int(a * sr):int((a + 0.4) * sr)] = 0.0
    x = 0.3 * carrier * syllables * level + 1e-3 * np.random.default_rng(0).standard_normal(len(t))
    y, info = dynamics.leveler(x, sr)
    assert info["applied"] and info["moves_over_1db_per_100ms_per_s"] < 0.3
    win, hop = int(0.05 * sr), int(0.01 * sr)
    rx = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(x, win)[::hop] ** 2, axis=1))
    ry = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(y, win)[::hop] ** 2, axis=1))
    g = 20 * np.log10(ry / rx)
    assert np.all(np.abs(g) <= 3.05)
    i_step = int(4.2 / 0.01)                                  # the quiet phrase starts after the pause at 3.8-4.2 s
    assert abs(g[i_step + 10] - g[i_step]) < 0.6              # nothing happens inside the first 100 ms
    assert g[i_step + 150] - g[i_step - 60] > 1.0             # but the quiet phrase is lifted within 1.5 s
    assert np.std(g[60:170]) < 0.4 and np.std(g[640:760]) < 0.4   # syllables are not chased


def test_k_weighting_matches_the_standard_table():
    pre_b, pre_a, rlb_b, rlb_a = dynamics.k_weighting_coefficients(48000)
    assert np.allclose(pre_b, dynamics._PRE_B, atol=1e-6) and np.allclose(pre_a, dynamics._PRE_A, atol=1e-6)
    assert np.allclose(rlb_b, dynamics._RLB_B, atol=1e-6) and np.allclose(rlb_a, dynamics._RLB_A, atol=1e-6)


def test_loudness_native_rate_reference_tone():
    """A 997 Hz sine at -20 dBFS peak reads -23.0 LUFS in one channel, at
    any sample rate."""
    for sr in (44100, 22050, 96000):
        t = np.arange(int(10 * sr)) / sr
        x = 10 ** (-20 / 20) * np.sin(2 * np.pi * 997 * t)
        assert abs(dynamics.integrated_loudness(x, sr) - (-23.01)) < 0.15, sr


def test_frame_rms_running_sum_matches_direct():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(20000) * np.linspace(0.1, 1.0, 20000)
    sr = 8000
    centres, levels = dynamics.frame_rms_db(x, sr, win_s=0.05, hop_s=0.01)
    win, hop = int(0.05 * sr), int(0.01 * sr)
    direct = [10 * np.log10(np.mean(x[i:i + win] ** 2)) for i in range(0, len(x) - win + 1, hop)]
    assert np.allclose(levels, direct, atol=1e-6) and len(centres) == len(direct)
