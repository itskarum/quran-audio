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


def test_recitation_reference_is_darker_than_speech():
    """Tajweed recitation carries far less energy above 1.5 kHz than
    conversational speech; the built-in reference must say so."""
    centres = eq.third_octave_centres(125.0, 8000.0)
    rec = eq.reference_recitation_db(centres)
    spk = eq.reference_ltass_db(centres)
    anchor = (centres >= 400) & (centres <= 800)
    rec, spk = rec - rec[anchor].mean(), spk - spk[anchor].mean()
    assert np.all(rec[centres >= 2000] < spk[centres >= 2000] - 3.0)
    assert np.all(rec[centres >= 4000] < spk[centres >= 4000] - 8.0)
    assert np.all(rec[centres <= 315] > spk[centres <= 315] + 3.0)
    assert np.allclose(eq.resolve_reference("recitation", centres), eq.reference_recitation_db(centres))
    assert np.allclose(eq.resolve_reference("speech", centres), eq.reference_ltass_db(centres))
    pair = (centres, rec)
    assert np.allclose(eq.resolve_reference(pair, centres), rec)


def test_tonal_balance_caps(sr):
    """A dull, band-limited transfer: boosts are capped above 5 kHz, absent
    where the voice is 20 dB down, and never applied below 150 Hz."""
    n_fft = 1408 if sr == 44100 else 704
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # a voice spectrum that falls 12 dB/octave above 1 kHz and is thin below 200 Hz
    psd = np.where(freqs > 1000, (1000 / np.maximum(freqs, 1)) ** 4, 1.0) * np.where(freqs < 200, 0.05, 1.0)
    taps, bands = eq.design_tonal_balance(psd, freqs, sr, 100.0, 0.45 * sr, strength=1.0, max_db=9.0)
    assert taps is not None and len(bands) >= 8
    for b in bands:
        if b["centre_hz"] > 5000:
            assert b["correction_db"] <= 1.5 + 1e-9
        if b["centre_hz"] < 150:
            assert b["correction_db"] <= 0.0
        if b["centre_hz"] < 300:
            assert b["correction_db"] <= 2.0 + 1e-9
        if b["measured_db"] < -20.0:
            assert b["correction_db"] <= 0.0


def test_reference_from_file(tmp_path, voice, sr):
    from quran_audio import audio_io
    path = tmp_path / "ref.wav"
    audio_io.save(path, voice, sr)
    centres, levels = eq.reference_from_file(path)
    assert len(centres) == len(levels) and np.isfinite(levels).sum() >= 8
    taps, bands = eq.design_tonal_balance(np.ones(705), np.fft.rfftfreq(1408, 1 / 44100), 44100, 100.0, 12000.0,
                                          reference=str(path))
    assert taps is not None and bands
