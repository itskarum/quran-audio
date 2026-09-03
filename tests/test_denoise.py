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
    # The synthetic voice has exact-zero pauses, so the noise profile comes
    # from the valley floor, whose reach on a perfectly periodic voice is
    # bounded by the analysis window's leakage (about -45 dB). What the
    # denoiser touches is therefore 30 dB down or more: inaudible, and the
    # SNR-adaptive floor keeps it there.
    from quran_audio.pipeline import adaptive_floor_db
    an = analysis.analyze(voice, sr, hum_search=False)
    y, _ = denoise.denoise(voice, sr, an, denoise.DenoiseSettings(floor_db=adaptive_floor_db(-18.0, -8.0, an.snr_db)))
    assert 20 * np.log10(np.linalg.norm(y - voice) / np.linalg.norm(voice)) < -30


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


def test_muted_gaps_still_denoised(voice, sr):
    """An edited transfer (digital silence between phrases) must get the
    same treatment as the intact file: noise inside speech goes down, the
    voice stays, and the report says what happened."""
    from conftest import at_snr, white, si_sdr_db
    noisy = voice + at_snr(voice, white(len(voice), 2), 10)
    ok = analysis.analyze(noisy, sr, hum_search=False)
    edited = noisy.copy()
    for a, b in ok.pause_ranges:
        edited[a:b] = 0.0
    an = analysis.analyze(edited, sr, hum_search=False)
    assert not an.anchors_reliable and an.noise_measurable
    y, info = denoise.denoise(edited, sr, an, denoise.DenoiseSettings(floor_db=-18.0))
    y_ok, _ = denoise.denoise(noisy, sr, ok, denoise.DenoiseSettings(floor_db=-18.0))
    assert info["notes"]
    env = np.sqrt(np.convolve(voice * voice, np.ones(int(0.02 * sr)) / int(0.02 * sr), "same"))
    act = env > 10 ** (-40 / 20)

    def in_speech_noise(out):
        a = np.dot(out, voice) / np.dot(voice, voice)
        return 10 * np.log10(np.mean((out - a * voice)[act] ** 2)), 20 * np.log10(a)

    n_ed, v_ed = in_speech_noise(y)
    n_ok, v_ok = in_speech_noise(y_ok)
    before = 10 * np.log10(np.mean((edited - voice)[act] ** 2))
    assert before - n_ed >= 3.0, (before, n_ed)
    assert abs(n_ed - n_ok) < 2.5 and abs(v_ed - v_ok) < 1.5, (n_ed, n_ok, v_ed, v_ok)
    assert si_sdr_db(voice, y) >= si_sdr_db(voice, edited) + 2.0


def test_do_no_harm_at_high_snr(voice, sr):
    """With the SNR-adaptive floor the denoiser must not cost more than it
    removes on a clean transfer."""
    from conftest import at_snr, white, si_sdr_db
    from quran_audio.pipeline import adaptive_floor_db
    for snr in (25.0, 30.0, 40.0):
        noisy = voice + at_snr(voice, white(len(voice), 3), snr)
        an = analysis.analyze(noisy, sr, hum_search=False)
        y, _ = denoise.denoise(noisy, sr, an, denoise.DenoiseSettings(floor_db=adaptive_floor_db(-18.0, -8.0, an.snr_db)))
        assert si_sdr_db(voice, y) >= si_sdr_db(voice, noisy) - 0.5, snr
