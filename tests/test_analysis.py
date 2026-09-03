import numpy as np

from conftest import at_snr, make_voice, white
from quran_audio import analysis


def test_noise_floor_snr_pauses_bandwidth(voice, sr):
    noisy = voice + at_snr(voice, white(len(voice)), 20)
    an = analysis.analyze(noisy, sr)
    assert 14 < an.snr_db < 26
    assert len(an.pause_ranges) >= 2 and an.pause_frames is not None and len(an.pause_frames) == len(an.pause_ranges)
    assert 0.05 < an.pause_fraction < 0.4 and an.speech_fraction > 0.5
    assert an.noise_psd.shape == an.speech_psd.shape == an.psd_freqs.shape
    assert an.clipped_runs == 0
    d = an.to_dict()
    assert d["pause_count"] == len(an.pause_ranges) and "hum" in d


def test_bandwidth_edge_of_bandlimited_voice(voice, sr):
    from scipy.signal import butter, sosfiltfilt
    limited = sosfiltfilt(butter(8, 3000, fs=sr, output="sos"), voice)
    noisy = limited + at_snr(limited, white(len(voice)), 25)
    an = analysis.analyze(noisy, sr, hum_search=False)
    assert 2500 < an.bandwidth_hz < 4200
    assert an.low_edge_hz < 150


def test_clipping_detection(voice):
    clipped = np.clip(voice * 3, -0.5, 0.5)
    runs, peak = analysis.count_clipped_runs(clipped)
    assert peak == 0.5 and runs > 50
    assert analysis.count_clipped_runs(voice)[0] == 0


def test_hum_detected_with_harmonics(voice, sr):
    t = np.arange(len(voice)) / sr
    hum = 0.02 * np.sin(2 * np.pi * 50.3 * t) + 0.008 * np.sin(2 * np.pi * 100.6 * t + 1) + 0.005 * np.sin(2 * np.pi * 150.9 * t + 2)
    noisy = voice + hum + at_snr(voice, white(len(voice)), 45)
    an = analysis.analyze(noisy, sr)
    assert an.hum.detected
    assert abs(an.hum.fundamental_hz - 50.3) < 0.15
    freqs = [h.freq_hz for h in an.hum.harmonics]
    assert any(abs(f - 100.6) < 0.5 for f in freqs)
    assert all(h.prominence_db >= 6 for h in an.hum.harmonics)


def test_sustained_voice_is_not_hum(sr):
    """A reciter holding notes near 100 Hz produces a comb at multiples of
    50 Hz in the whole-file spectrum; it must not be mistaken for hum."""
    v = make_voice(sr, 14.0, f0=100.0, vibrato=0.0)
    noisy = v + at_snr(v, white(len(v)), 40)
    an = analysis.analyze(noisy, sr)
    assert not an.hum.detected


def test_short_signal_hum_skipped(sr):
    rep = analysis.detect_hum(np.zeros(sr), sr)
    assert not rep.detected and "short" in rep.note


def test_sustained_voice_with_pauses_is_not_hum(sr):
    """Same trap, but with enough breath pauses for the confirmation step:
    the comb is absent while the reciter is silent, so it is not hum."""
    v = make_voice(sr, 24.0, f0=100.0, vibrato=0.0)
    noisy = v + at_snr(v, white(len(v)), 40)
    an = analysis.analyze(noisy, sr)
    assert len(an.pause_ranges) >= 4
    assert not an.hum.detected


def test_hum_confirmed_in_pauses(sr):
    v = make_voice(sr, 24.0, f0=120.0)
    t = np.arange(len(v)) / sr
    noisy = v + 0.01 * np.sin(2 * np.pi * 60.0 * t) + 0.004 * np.sin(2 * np.pi * 180.0 * t) + at_snr(v, white(len(v)), 40)
    an = analysis.analyze(noisy, sr)
    assert an.hum.detected and abs(an.hum.fundamental_hz - 60.0) < 0.15
    assert "pauses" in an.hum.note


def test_muted_gaps_do_not_poison_the_noise_profile(voice, sr):
    """An edited transfer with digital silence between phrases must not
    hand the denoiser a profile made of quiet voice."""
    from conftest import at_snr, white
    noisy = voice + at_snr(voice, white(len(voice)), 20)
    ok = analysis.analyze(noisy, sr, hum_search=False)
    edited = noisy.copy()
    for a, b in ok.pause_ranges:
        edited[a:b] = 0.0
    an = analysis.analyze(edited, sr, hum_search=False)
    assert not an.anchors_reliable and any("valley floor" in n for n in an.notes)
    band = (ok.psd_freqs >= 200) & (ok.psd_freqs <= 2000)
    diff = 10 * np.log10(an.noise_psd[band].mean() / ok.noise_psd[band].mean())
    assert abs(diff) < 4.0, diff
    assert an.snr_db > ok.snr_db - 4.0
