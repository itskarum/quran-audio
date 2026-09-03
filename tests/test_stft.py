import numpy as np
import pytest

from quran_audio.stft import STFT, frame_length_for


@pytest.mark.parametrize("n_fft", [512, 704, 1536])
@pytest.mark.parametrize("n", [1, 100, 511, 512, 5123])
def test_identity_reconstruction(n_fft, n):
    x = np.random.default_rng(0).standard_normal(n)
    st = STFT(n_fft)
    y = st.process(x, lambda spec, m0: spec, block_frames=3)
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-10


def test_gain_scales_output():
    x = np.random.default_rng(1).standard_normal(4000)
    y = STFT(256).process(x, lambda spec, m0: 0.5 * spec)
    assert np.allclose(y, 0.5 * x, atol=1e-10)


def test_geometry():
    st = STFT(512)
    assert st.hop == 128 and st.n_bins == 257
    assert st.n_frames(1) == (384 + 0) // 128 + 1
    spec_blocks = list(st.iter_blocks(np.zeros(3000), block_frames=5))
    assert sum(b.shape[0] for _, b in spec_blocks) == st.n_frames(3000)
    with pytest.raises(ValueError):
        STFT(500, 3)
    assert frame_length_for(44100) == 1408 and frame_length_for(48000) == 1536 and frame_length_for(16000) == 512
