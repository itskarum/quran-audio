import numpy as np

from conftest import at_snr, white
from quran_audio import analysis, eq


def test_reference_shape():
    f = np.array([63, 100, 250, 400, 630, 1000, 2000, 4000, 8000.0])
    r = eq.reference_ltass_db(f)
    assert r[3] == 0 and r[4] == 0
    assert np.all(np.diff(r[:4]) > 0) and np.all(np.diff(r[4:]) < 0)


def test_tonal_balance_bounded_zero_delay(voice, sr):
    an = analysis.analyze(voice + at_snr(voice, white(len(voice)), 30), sr, hum_search=False)
    taps, bands = eq.design_tonal_balance(an.speech_psd, an.psd_freqs, sr, an.low_edge_hz, an.bandwidth_hz,
                                          strength=0.5, max_db=6.0)
    assert taps is not None and len(taps) % 2 == 1 and len(bands) >= 6
    assert all(abs(b["correction_db"]) <= 3.0 + 1e-9 for b in bands)
    assert all(b["correction_db"] <= 0 for b in bands if b["centre_hz"] < 150)
    y = eq.apply_fir(voice, taps)
    assert len(y) == len(voice)
    lag = np.argmax(np.correlate(y[:sr], voice[:sr], mode="full")) - (sr - 1)
    assert lag == 0
    assert eq.design_tonal_balance(an.speech_psd, an.psd_freqs, sr, 100, 200, 0.5, 6)[0] is None


def test_highpass_lowpass_and_cutoff():
    sr = 22050
    t = np.arange(sr * 2) / sr
    low, mid = np.sin(2 * np.pi * 20 * t), np.sin(2 * np.pi * 200 * t)
    y = eq.highpass(low + mid, sr, 40.0)
    assert 20 * np.log10(np.std(y - mid) / np.std(low)) < -20 or np.std(y - mid) < 0.15
    assert abs(np.std(eq.highpass(mid, sr, 40.0)) / np.std(mid) - 1) < 0.05
    hi = np.sin(2 * np.pi * 8000 * t)
    assert np.std(eq.lowpass(hi, sr, 4000.0)) / np.std(hi) < 0.05
    assert eq.lowpass(hi, sr, 11000.0) is hi
    assert eq.rumble_cutoff(0) == 40 and eq.rumble_cutoff(94) == 56.4 and eq.rumble_cutoff(500) == 80
