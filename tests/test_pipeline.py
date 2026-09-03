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


def test_speed_correction_is_opt_in_and_reported(sr):
    from conftest import make_voice, at_snr, white
    v = make_voice(sr, 24.0)
    t = np.arange(len(v)) / sr
    noisy = v + 0.01 * np.sin(2 * np.pi * 51.0 * t) + 0.004 * np.sin(2 * np.pi * 102.0 * t) + at_snr(v, white(len(v)), 40)
    plain = enhance_signal(noisy, sr, make_settings("gentle", {"target_lufs": None, "denoise": "off"}))
    assert plain.report["speed_correction"] is None and plain.samples.shape[0] == len(noisy)
    fixed = enhance_signal(noisy, sr, make_settings("gentle", {"target_lufs": None, "denoise": "off", "speed_correct": True}))
    rep = fixed.report["speed_correction"]
    assert rep["applied"] and abs(rep["speed_error_pct"] - 2.0) < 0.6
    assert abs(fixed.samples.shape[0] / len(noisy) - 1.02) < 0.005      # ran 2 % fast: slowed down, so longer
    assert not fixed.report["length_preserved"]


def test_provenance_written_to_output(tmp_path, voice, sr):
    from quran_audio import audio_io
    from quran_audio.pipeline import enhance_file
    src = tmp_path / "in.flac"
    audio_io.save(src, voice, sr, tags={"title": "Al-Fatiha", "artist": "reciter"})
    rep = enhance_file(src, tmp_path / "out.flac", make_settings("gentle", {"target_lufs": None}))
    tags = audio_io.load(tmp_path / "out.flac").tags
    assert tags["title"] == "Al-Fatiha" and tags["artist"] == "reciter"
    assert tags["software"].startswith("quran-audio")
    import json
    summary = json.loads(tags["comment"])
    assert summary["preset"] == "gentle" and "voice_band_db" in summary
    assert rep["output"]["dithered"] is False          # 24-bit default: no dither


def test_breath_attenuation_is_opt_in_and_leaves_the_voice(sr):
    from conftest import make_voice
    v = make_voice(sr, 24.0)
    noisy = v + at_snr(v, white(len(v), 2), 45)
    rng = np.random.default_rng(9)
    sos = butter(4, [500, 4000], btype="band", fs=sr, output="sos")
    inserted = []
    for start in (4.15, 9.15, 14.15, 19.15):
        a, b = int(start * sr), int((start + 0.3) * sr)
        noisy[a:b] += sosfiltfilt(sos, rng.standard_normal(b - a)) * 10 ** (-32 / 20) * np.hanning(b - a)
        inserted.append((a, b))
    plain = enhance_signal(noisy, sr, make_settings("standard", {"target_lufs": None}))
    assert "breath" not in plain.report["channels"][0]
    soft = enhance_signal(noisy, sr, make_settings("standard", {"target_lufs": None, "breath_db": -6.0}))
    info = soft.report["channels"][0]["breath"]
    assert info["count"] >= 3 and info["attenuation_db"] == -6.0
    for a, b in inserted[:3]:
        mid = slice(a + int(0.08 * sr), b - int(0.08 * sr))
        drop = 20 * np.log10(np.std(soft.samples[mid, 0]) / np.std(plain.samples[mid, 0]))
        assert abs(drop + 6.0) < 1.5, drop
    voiced = slice(int(1.0 * sr), int(3.5 * sr))
    assert np.array_equal(soft.samples[voiced], plain.samples[voiced])
    # never more than -12 dB, whatever is asked
    deep = enhance_signal(noisy, sr, make_settings("standard", {"target_lufs": None, "breath_db": -30.0}))
    assert deep.report["channels"][0]["breath"]["attenuation_db"] == -12.0
