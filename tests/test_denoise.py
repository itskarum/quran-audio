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
    y, _ = denoise.denoise(voice, sr, an, denoise.DenoiseSettings(floor_db=adaptive_floor_db(-18.0, -12.0, an.snr_db)))
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
    assert abs(n_ed - n_ok) < 2.5 and abs(v_ed - v_ok) < 3.0, (n_ed, n_ok, v_ed, v_ok)
    assert si_sdr_db(voice, y) >= si_sdr_db(voice, edited) + 2.0


def test_do_no_harm_at_high_snr(voice, sr):
    """With the SNR-adaptive floor the denoiser must not cost more than it
    removes on a clean transfer."""
    from conftest import at_snr, white, si_sdr_db
    from quran_audio.pipeline import adaptive_floor_db
    for snr in (25.0, 30.0, 40.0):
        noisy = voice + at_snr(voice, white(len(voice), 3), snr)
        an = analysis.analyze(noisy, sr, hum_search=False)
        y, _ = denoise.denoise(noisy, sr, an, denoise.DenoiseSettings(floor_db=adaptive_floor_db(-18.0, -12.0, an.snr_db)))
        assert si_sdr_db(voice, y) >= si_sdr_db(voice, noisy) - 0.5, snr


def _reverberant(voice, sr, rt60=1.5, seed=11):
    """Voice through a synthetic hall: exponentially decaying noise tail
    with a direct path at about 0 dB direct-to-reverberant ratio."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    h = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    h /= np.sqrt(np.sum(h ** 2))
    h[0] = 1.0
    return fftconvolve(voice, h)[:len(voice)]


def test_reverb_tail_preserved(voice, sr):
    """After a phrase ends the room decays into the noise; the denoiser must
    let it, instead of gating it to the floor within a few hundred ms."""
    from conftest import at_snr, white
    from quran_audio.stft import STFT, frame_length_for
    wet = _reverberant(voice, sr)
    noisy = wet + at_snr(wet, white(len(wet), 12), 30)
    an = analysis.analyze(noisy, sr, hum_search=False)
    assert 15.0 <= an.decay_db_per_s <= 90.0, an.decay_db_per_s       # RT60 1.5 s is about 40 dB/s
    y_on, info = denoise.denoise(noisy, sr, an, denoise.DenoiseSettings(floor_db=-18.0, tail_preserve=True))
    y_off, _ = denoise.denoise(noisy, sr, an, denoise.DenoiseSettings(floor_db=-18.0, tail_preserve=False))
    assert info["tail_preserve"]
    st = STFT(frame_length_for(sr))
    fr = st.freqs(sr)
    band = (fr >= 200) & (fr < 4000)

    def energy(x):
        return np.concatenate([p[:, band].mean(axis=1) for _, p in st.power_blocks(x)])

    e_in, e_on, e_off = energy(noisy), energy(y_on), energy(y_off)
    # at RT60 1.5 s and 30 dB SNR the tail is only ~5 dB above the noise at
    # the (threshold-defined) offset and reaches the noise ~150 ms later, so
    # the comparison is made early in the decay
    for ms, min_on, min_gain in ((50, -4.0, 3.0), (100, -7.0, 4.0)):
        k = int(ms / 1000 * sr / st.hop)
        cut_on, cut_off = [], []
        for m in an.offset_frames:
            if m + k < len(e_in):
                cut_on.append(10 * np.log10(e_on[m + k] / e_in[m + k]))
                cut_off.append(10 * np.log10(e_off[m + k] / e_in[m + k]))
        assert len(cut_on) >= 2
        assert np.median(cut_on) > min_on, (ms, np.median(cut_on))
        assert np.median(cut_off) < np.median(cut_on) - min_gain, (ms, np.median(cut_off), np.median(cut_on))
