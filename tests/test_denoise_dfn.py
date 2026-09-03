import hashlib
import io
import zipfile

import pytest

from quran_audio import denoise_dfn


def _archive(with_config=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if with_config:
            zf.writestr("DeepFilterNet3/config.ini", "[df]\n")
            zf.writestr("DeepFilterNet3/checkpoints/model_1.ckpt.best", b"weights")
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_model_verifies_checksum_and_extracts(tmp_path, monkeypatch):
    data = _archive()
    monkeypatch.setattr(denoise_dfn.urllib.request, "urlopen", lambda url, timeout=0: _Resp(data))
    with pytest.raises(RuntimeError, match="checksum"):
        denoise_dfn.fetch_model(tmp_path, sha256="0" * 64)
    path = denoise_dfn.fetch_model(tmp_path, sha256=hashlib.sha256(data).hexdigest())
    assert (path / "config.ini").is_file() and denoise_dfn.model_dir(path) == path
    assert denoise_dfn.fetch_model(tmp_path, sha256="irrelevant-when-cached") == path


def test_model_dir_and_cache_env(tmp_path, monkeypatch):
    assert denoise_dfn.model_dir(tmp_path / "missing") is None
    monkeypatch.setenv("QURAN_AUDIO_CACHE", str(tmp_path / "c"))
    assert denoise_dfn.cache_dir() == tmp_path / "c"


def test_unavailable_message_without_torch(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("torch", "libdf"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(denoise_dfn.DFNUnavailable, match="fetch-models"):
        denoise_dfn.DFNDenoiser()
