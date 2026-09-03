"""The fidelity self-check that every run carries in its report."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from conftest import at_snr, make_voice, white
from quran_audio import fidelity
from quran_audio.pipeline import enhance_signal, make_settings


def test_fidelity_measures_and_warns(sr):
    voice = make_voice(sr, 30.0)
    noisy = voice + at_snr(voice, white(len(voice), 2), 20)
    f = fidelity.measure(noisy, noisy, sr)
    assert f["voice_band_retention_db"] == 0.0 and f["projection_db"] == 0.0 and not f["warnings"]
    assert len(f["bands_hz"]) == len(f["speech_retention_db"]) == len(f["pause_reduction_db"])
    assert f["onsets"] >= 3 and f["onset_retention_db"] == 0.0
    # carve 4 dB out of 1-3 kHz: the voice-band bound must fire
    sos = butter(2, [1000, 3000], btype="band", fs=sr, output="sos")
    damaged = noisy - (1 - 10 ** (-4 / 20)) * sosfiltfilt(sos, noisy)
    f2 = fidelity.measure(noisy, damaged, sr)
    assert any("voice band" in w for w in f2["warnings"])
    # halve the whole signal: the overall level bound must fire
    f3 = fidelity.measure(noisy, 0.5 * noisy, sr)
    assert any("overall voice level" in w for w in f3["warnings"])
    assert "voice band" in fidelity.summary(f2)


def test_pipeline_report_carries_fidelity(sr):
    voice = make_voice(sr, 20.0)
    noisy = voice + at_snr(voice, white(len(voice), 3), 20)
    res = enhance_signal(noisy, sr, make_settings("standard", {"target_lufs": None}))
    f = res.report["fidelity"]
    assert f is not None and f["voice_band_retention_db"] > -1.0 and f["in_speech_repairs"] == 0
    assert not [w for w in res.report["notes"] if w.startswith("fidelity:")], res.report["notes"]
