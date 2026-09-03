import numpy as np
from scipy.signal import butter, sosfiltfilt

from conftest import at_snr, make_voice, white
from quran_audio import analysis, hum


def tone_level_db(sig, f, sr):
    t = np.arange(len(sig)) / sr
    w = np.hanning(len(sig))
    return 20 * np.log10(np.abs(np.dot(sig * w, np.exp(-2j * np.pi * f * t))) / np.sum(w) * 2 + 1e-12)


def test_hum_removed_voice_kept(voice, sr):
    t = np.arange(len(voice)) / sr
    hum_sig = 0.02 * np.sin(2 * np.pi * 50.3 * t) + 0.008 * np.sin(2 * np.pi * 100.6 * t + 1)
    noisy = voice + hum_sig + at_snr(voice, white(len(voice)), 45)
    an = analysis.analyze(noisy, sr)
    y, applied = hum.remove_hum(noisy, sr, an.hum, pause_ranges=an.pause_ranges)
    assert len(y) == len(noisy) and len(applied) >= 2
    for f in (50.3, 100.6):
        assert tone_level_db(noisy, f, sr) - tone_level_db(y, f, sr) > 40
    sos = butter(4, [300, 3000], btype="band", fs=sr, output="sos")
    removed_voice = sosfiltfilt(sos, (noisy - y) - hum_sig)
    assert 20 * np.log10(np.std(removed_voice) / np.std(sosfiltfilt(sos, voice))) < -35


def test_noop_without_detection(voice, sr):
    y, applied = hum.remove_hum(voice, sr, analysis.HumReport(False, 0.0, []))
    assert y is voice and applied == []


def test_tracker_does_not_ring_into_pauses(sr):
    """A voice harmonic sitting exactly on a hum harmonic (250 Hz) must not
    leave tonal tails in the breath pauses after removal."""
    v = make_voice(sr, 24.0, f0=125.0)
    t = np.arange(len(v)) / sr
    hum_sig = 0.01 * np.sin(2 * np.pi * 50.0 * t) + 0.004 * np.sin(2 * np.pi * 250.0 * t + 1)
    noisy = v + hum_sig + at_snr(v, white(len(v)), 50)
    an = analysis.analyze(noisy, sr)
    assert an.hum.detected and any(abs(h.freq_hz - 250.0) < 0.5 for h in an.hum.harmonics)
    y, applied = hum.remove_hum(noisy, sr, an.hum, pause_ranges=an.pause_ranges)
    assert all(a["method"] == "track" for a in applied)
    sos = butter(4, [230, 270], btype="band", fs=sr, output="sos")
    for a, b in an.pause_ranges[1:4]:
        seg = slice(a + int(0.1 * sr), b - int(0.1 * sr))
        before = np.std(sosfiltfilt(sos, noisy)[seg])
        after = np.std(sosfiltfilt(sos, y)[seg])
        assert after < 0.25 * before          # hum gone from the pause, nothing added
    assert np.array_equal(hum.remove_hum(v, sr, analysis.HumReport(False, 0, []))[0], v)


def test_tracker_exact_on_stationary_tone():
    sr = 8000
    t = np.arange(sr * 6) / sr
    x = 0.1 * np.sin(2 * np.pi * 60.2 * t + 0.3) + 0.001 * np.random.default_rng(0).standard_normal(len(t))
    y, refined = hum.track_and_subtract(x, sr, np.array([60.2]), np.array([2.0]))
    assert tone_level_db(x, 60.2, sr) - tone_level_db(y, 60.2, sr) > 50
    # a 0.3 Hz error in the detected frequency is refined away from the phase drift
    y2, refined2 = hum.track_and_subtract(x, sr, np.array([60.5]), np.array([2.0]))
    assert abs(refined2[0] - 60.2) < 0.02
    assert tone_level_db(x, 60.2, sr) - tone_level_db(y2, 60.2, sr) > 40


def test_buzz_line_above_two_kilohertz_and_speed_report(sr):
    """A transformer buzz line at 2.5 kHz counts (with the stricter 10 dB
    requirement), and the mains line implies the transfer speed."""
    v = make_voice(sr, 24.0, f0=140.0)
    t = np.arange(len(v)) / sr
    lines = 0.01 * np.sin(2 * np.pi * 51.0 * t) + 0.004 * np.sin(2 * np.pi * 102.0 * t + 1) + 0.003 * np.sin(2 * np.pi * 2550.0 * t + 2)
    noisy = v + lines + at_snr(v, white(len(v)), 45)
    an = analysis.analyze(noisy, sr)
    assert an.hum.detected and abs(an.hum.fundamental_hz - 51.0) < 0.3
    assert any(abs(h.freq_hz - 2550.0) < 3.0 for h in an.hum.harmonics), [h.freq_hz for h in an.hum.harmonics]
    d = an.hum.to_dict()
    assert d["nominal_hz"] == 50.0 and abs(d["speed_error_pct"] - 2.0) < 0.6 and d["line_width_hz"] is not None
    y, applied = hum.remove_hum(noisy, sr, an.hum, pause_ranges=an.pause_ranges)
    assert tone_level_db(noisy, 2550.0, sr) - tone_level_db(y, 2550.0, sr) > 20


def test_two_combs_speed_reading_follows_the_mains_line(sr):
    """Hum recorded on the tape plays back shifted with the transfer speed
    (50.4 Hz: 0.8 % fast); hum picked up at the transfer sits at the mains
    frequency (49.96 Hz) and shows in its higher harmonics. The speed
    reading must follow the mains line itself, and the report says that
    two combs are present."""
    v = make_voice(sr, 24.0, f0=140.0)
    t = np.arange(len(v)) / sr
    tape = 0.01 * np.sin(2 * np.pi * 50.4 * t) + 0.004 * np.sin(2 * np.pi * 100.8 * t + 1)
    transfer = 0.004 * np.sin(2 * np.pi * 3 * 49.96 * t + 2) + 0.003 * np.sin(2 * np.pi * 5 * 49.96 * t + 3)
    noisy = v + tape + transfer + at_snr(v, white(len(v)), 45)
    an = analysis.analyze(noisy, sr)
    assert an.hum.detected and abs(an.hum.fundamental_hz - 50.4) < 0.15, an.hum.fundamental_hz
    d = an.hum.to_dict()
    assert abs(d["speed_error_pct"] - 0.8) < 0.3, d["speed_error_pct"]
    assert "two fundamentals" in d["note"] and "49.9" in d["note"], d["note"]
    y, applied = hum.remove_hum(noisy, sr, an.hum, pause_ranges=an.pause_ranges)
    for f, drop in ((50.4, 15), (100.8, 15), (149.88, 8), (249.8, 8)):   # the voice's own energy floors the measurement near 150 Hz
        assert tone_level_db(noisy, f, sr) - tone_level_db(y, f, sr) > drop, f
