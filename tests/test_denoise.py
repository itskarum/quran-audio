import numpy as np
import pytest

from conftest import at_snr, pause_slice, white
from quran_audio import analysis, denoise


def _level(x):
    return 20 * np.log10(np.std(x) + 1e-12)


@pytest.mark.parametrize("fusion", ["soft", "spp", "xi", "none"])
def test_noise_reduced_in_pauses(voice, sr, fusion):
    noisy = voice + at_snr(voice, white(len(voice)), 15)
    an = analysis.analyze(noisy, sr, hum_search=False)
    y, info = denoise.denoise(noisy, sr, an, denoise.DenoiseSettings(floor_db=-18, fusion=fusion))
    assert len(y) == len(noisy) and info["pause_anchors"] >= 2
    pz = pause_slice(sr, 0)
    assert _level(noisy[pz]) - _level(y[pz]) > (8 if fusion == "none" else 12)
    # the voice is still there: correlation with the clean reference stays high
    assert np.corrcoef(voice, y)[0, 1] > 0.9


def test_clean_input_nearly_unchanged(voice, sr):
    an = analysis.analyze(voice, sr, hum_search=False)
    y, _ = denoise.denoise(voice, sr, an)
    assert 20 * np.log10(np.linalg.norm(y - voice) / np.linalg.norm(voice)) < -40


def test_deterministic_and_profile_fallback(voice, sr):
    noisy = voice + at_snr(voice, white(len(voice)), 12)
    y1, _ = denoise.denoise(noisy, sr)
    y2, _ = denoise.denoise(noisy, sr)
    assert np.array_equal(y1, y2)
    pz = pause_slice(sr, 0)
    assert _level(noisy[pz]) - _level(y1[pz]) > 8


def test_gain_floor_deeper_above_bandwidth():
    st = denoise.SpectralDenoiser(44100, 1408, np.ones(705), denoise.DenoiseSettings(floor_db=-18, hf_floor_db=-40, bandwidth_hz=4000))
    freqs = np.fft.rfftfreq(1408, 1 / 44100)
    fl = 20 * np.log10(st.floor)
    assert abs(fl[np.searchsorted(freqs, 1000)] + 18) < 0.01
    assert abs(fl[np.searchsorted(freqs, 8000)] + 40) < 0.01
    with pytest.raises(ValueError):
        denoise.SpectralDenoiser(44100, 1408, np.ones(10), denoise.DenoiseSettings())
