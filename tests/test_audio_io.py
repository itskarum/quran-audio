import numpy as np
import pytest

from quran_audio import audio_io


def test_resample_unity_gain_and_length():
    sr = 44100
    t = np.arange(sr * 2) / sr
    x = 0.5 * np.sin(2 * np.pi * 1000 * t)
    y = audio_io.resample(x, sr, 48000)
    assert len(y) == audio_io.resampled_length(len(x), sr, 48000) == 96000
    assert abs(np.max(np.abs(y[2000:-2000])) / 0.5 - 1.0) < 1e-3
    z = audio_io.fit_length(audio_io.resample(y, 48000, sr), len(x))
    assert 20 * np.log10(np.max(np.abs(z[5000:-5000] - x[5000:-5000])) / 0.5) < -90
    assert audio_io.resample(x, sr, sr) is x


def test_fit_length_pads_and_trims():
    x = np.arange(5.0)
    assert len(audio_io.fit_length(x, 8)) == 8 and audio_io.fit_length(x, 8)[-1] == 0
    assert np.array_equal(audio_io.fit_length(x, 3), x[:3])


def test_to_mono_strategies():
    n = 1000
    x = np.sin(np.linspace(0, 50, n))
    assert audio_io.to_mono(x)[1] == "mono"
    assert audio_io.to_mono(np.stack([x, x * 0.001], 1))[1] == "channel0"
    assert audio_io.to_mono(np.stack([x * 0.001, x], 1))[1] == "channel1"
    assert audio_io.to_mono(np.stack([x, -x], 1))[1] == "mix-inverted"
    m, s = audio_io.to_mono(np.stack([x, x], 1))
    assert s == "mix" and np.allclose(m, x)
    assert audio_io.to_mono(np.stack([x, 2 * x], 1), "left")[1] == "left"
    assert np.allclose(audio_io.to_mono(np.stack([x, 2 * x], 1), "right")[0], 2 * x)
    with pytest.raises(ValueError):
        audio_io.to_mono(np.stack([x, x], 1), "bogus")


@pytest.mark.parametrize("ext,tol", [("wav", 1e-6), ("flac", 1e-6), ("mp3", 0.05), ("ogg", 0.05)])
def test_save_load_roundtrip(tmp_path, ext, tol):
    sr = 22050
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    p = tmp_path / f"t.{ext}"
    info = audio_io.save(p, x, sr)
    assert info["clipped_samples"] == 0 and abs(info["peak"] - 0.5) < 1e-6
    a = audio_io.load(p)
    assert a.sample_rate == sr and a.n_channels == 1
    assert abs(a.n_samples - len(x)) <= 2 * 1152          # codec framing slack
    n = min(len(x), a.n_samples)
    assert np.max(np.abs(a.samples[:n, 0] - x[:n])) < tol


def test_save_clips_and_counts(tmp_path):
    p = tmp_path / "hot.wav"
    info = audio_io.save(p, np.array([0.0, 1.5, -2.0, 0.2]), 8000)
    assert info["clipped_samples"] == 2
    assert np.max(np.abs(audio_io.load(p).samples)) <= 1.0


def test_load_errors(tmp_path):
    with pytest.raises(audio_io.DecodeError):
        audio_io.load(tmp_path / "missing.wav")
    bad = tmp_path / "bad.xyz"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(audio_io.DecodeError):
        audio_io.load(bad)
    with pytest.raises(audio_io.EncodeError):
        audio_io.save(tmp_path / "out.xyz", np.zeros(10), 8000)


def test_remove_dc():
    y, off = audio_io.remove_dc(np.ones(10) * 0.25)
    assert off == 0.25 and np.allclose(y, 0)
