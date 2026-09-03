import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from conftest import at_snr, si_sdr_db, white
from quran_audio import make_settings, enhance_signal, pipeline


def make_old_recording(voice, sr, seed=5):
    rng = np.random.default_rng(seed)
    limited = sosfiltfilt(butter(6, 4000, fs=sr, output="sos"), voice)
    noise = at_snr(limited, white(len(voice), seed), 14)
    t = np.arange(len(voice)) / sr
    hum = 0.01 * np.sin(2 * np.pi * 50 * t) + 0.004 * np.sin(2 * np.pi * 100 * t + 1)
    clicks = np.zeros_like(voice)
    for p in rng.integers(1000, len(voice) - 1000, 80):
        clicks[p:p + rng.integers(1, 5)] += rng.uniform(0.1, 0.4) * rng.choice([-1, 1])
    return limited, limited + noise + hum + clicks, noise + hum + clicks


def test_end_to_end_standard(voice, sr):
    ref, degraded, removed = make_old_recording(voice, sr)
    settings = make_settings("standard", {"residual": True, "denoise": "classical"})
    result = enhance_signal(degraded, sr, settings)
    r = result.report
    assert result.samples.shape == (len(degraded), 1) and result.sample_rate == sr
    assert r["length_preserved"] and r["denoise_backend"] == "classical"
    assert r["analysis_after"]["snr_db"] > r["analysis_before"]["snr_db"] + 6
    assert r["channels"][0]["declick"]["clicks_repaired"] > 20
    assert len(r["channels"][0]["hum_notches"]) >= 1
    assert r["loudness"]["true_peak_after_dbtp"] <= -1.0 + 0.05
    assert abs(r["loudness"]["loudness_after_lufs"] + 18) < 0.5
    # the residual is what was removed: it should look like the noise, not the voice
    res = result.residual[:, 0]
    assert res.shape == (len(degraded),)
    captured = np.dot(res, removed) / np.dot(removed, removed)      # share of the disturbance removed
    voice_leak = np.dot(res, ref) / np.dot(ref, ref)                # share of the voice removed
    assert captured > 0.6
    assert abs(voice_leak) < 0.35
    # before the EQ stage the waveform gets much closer to the reference;
    # the final output also improves despite the deliberate tonal change
    pre_eq = (degraded - degraded.mean()) - res
    assert si_sdr_db(ref, pre_eq) > si_sdr_db(ref, degraded) + 2
    assert si_sdr_db(ref, result.samples[:, 0]) > si_sdr_db(ref, degraded)


def test_presets_and_overrides():
    s = make_settings("strong", {"target_lufs": None, "channels": "keep"})
    assert s.preset == "strong" and s.decrackle and s.target_lufs is None and s.channels == "keep"
    assert make_settings("gentle").leveler is False
    with pytest.raises(ValueError):
        make_settings("loud")


def test_stereo_keep_and_output_rate(voice, sr):
    _, degraded, _ = make_old_recording(voice, sr)
    stereo = np.stack([degraded, degraded * 0.7], 1)
    settings = make_settings("gentle", {"channels": "keep", "output_sr": 32000, "denoise": "classical",
                                        "target_lufs": None, "tonal_balance": False})
    result = enhance_signal(stereo, sr, settings)
    assert result.samples.shape[1] == 2 and result.sample_rate == 32000
    assert result.samples.shape[0] == pipeline.resampled_length(len(degraded), sr, 32000)
    assert result.report["length_preserved"] and len(result.report["channels"]) == 2


def test_denoise_off_and_dfn_unavailable(voice, sr):
    _, degraded, _ = make_old_recording(voice, sr)
    result = enhance_signal(degraded[: 4 * sr], sr, make_settings("standard", {"denoise": "off", "leveler": False}))
    assert result.report["channels"][0]["denoise"] == {"backend": "off"}
    pytest.importorskip("numpy")
    try:
        import torch  # noqa: F401
        pytest.skip("torch installed; fallback path not exercised here")
    except ImportError:
        pass
    from quran_audio.denoise_dfn import DFNUnavailable
    auto = enhance_signal(degraded[: 4 * sr], sr, make_settings("standard", {"denoise": "auto", "leveler": False}))
    assert auto.report["denoise_backend"] == "classical"
    with pytest.raises(DFNUnavailable):
        enhance_signal(degraded[: 4 * sr], sr, make_settings("standard", {"denoise": "dfn"}))


def test_silent_and_tiny_inputs():
    from conftest import make_voice
    sr = 8000
    silent = np.zeros(sr * 3)
    r = enhance_signal(silent, sr, make_settings("standard", {"denoise": "classical"}))
    assert r.samples.shape == (len(silent), 1) and np.all(np.isfinite(r.samples)) and r.report["length_preserved"]
    tiny = make_voice(sr, 0.5)
    r2 = enhance_signal(tiny, sr, make_settings("strong", {"denoise": "classical", "residual": True}))
    assert r2.samples.shape == (len(tiny), 1) and r2.residual.shape == (len(tiny), 1)
    assert np.all(np.isfinite(r2.samples)) and r2.report["length_preserved"]
