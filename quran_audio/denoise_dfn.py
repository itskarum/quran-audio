"""Optional deep denoising backend: DeepFilterNet3 (Schroeter et al., 2023).

DeepFilterNet is a discriminative model: it estimates a complex-valued
filter per time-frequency point and applies it to the input, so like the
classical backend it can only attenuate what is there. It does not
synthesise speech, which is why it is acceptable here where generative
"restoration" models are not. The optional attenuation limit caps how
deep it may cut, keeping a natural residual on very old material.

Status: experimental and opt-in. Besides noise the model removes
reverberation (about 9 dB of room energy on a reverberant hall
recording), its level guard and the 30 s chunk crossfades have been
exercised in one environment only, and it is not part of the fidelity
bounds the classical backend is held to. If its deeper pause cleaning is
ever wanted, the sound way to use it is as a second opinion, applying its
gains only where the classical stage already says "noise", not as a
replacement.

The Python package on PyPI (deepfilternet 0.5.6) is unmaintained: its
audio-I/O helper imports torchaudio APIs that no longer exist, and its
metadata pins numpy < 2. We never use that helper, so when `df.io` fails
to import we register a stub in its place; the model code itself runs
fine on current torch and numpy. Model weights are fetched once from the
project's GitHub repository and verified against a pinned SHA-256.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import types
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .audio_io import fit_length, resample

MODEL_NAME = "DeepFilterNet3"
MODEL_URL = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3.zip"
MODEL_SHA256 = "49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2"
DFN_SR = 48000

INSTALL_HINT = (
    "DeepFilterNet backend not available. Install the optional extra:\n"
    "  pip install 'quran-audio[dfn]'   (CPU-only torch: add "
    "--extra-index-url https://download.pytorch.org/whl/cpu)\n"
    "then fetch the model once with:  quran-audio fetch-models"
)


class DFNUnavailable(RuntimeError):
    """torch / deepfilternet not installed, or the model is missing."""


def cache_dir() -> Path:
    env = os.environ.get("QURAN_AUDIO_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "quran-audio"


def model_dir(explicit: str | os.PathLike | None = None) -> Path | None:
    """Directory holding config.ini + checkpoints/, or None if not present."""
    candidates = [Path(explicit)] if explicit else [cache_dir() / MODEL_NAME]
    for c in candidates:
        if (c / "config.ini").is_file() and (c / "checkpoints").is_dir():
            return c
    return None


def fetch_model(dest: str | os.PathLike | None = None, force: bool = False,
                url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> Path:
    """Download and verify the model archive, extract it, return its dir."""
    root = Path(dest) if dest else cache_dir()
    target = root / MODEL_NAME
    if model_dir(target) and not force:
        return target
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quran-audio-model-") as tmp:
        zpath = Path(tmp) / "model.zip"
        with urllib.request.urlopen(url, timeout=120) as resp, open(zpath, "wb") as fh:
            digest = hashlib.sha256()
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                fh.write(chunk)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f"model archive checksum mismatch: got {digest.hexdigest()}, expected {sha256}")
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                if member.startswith("/") or ".." in Path(member).parts:
                    raise RuntimeError(f"unsafe path in model archive: {member!r}")
            zf.extractall(root)
    if not model_dir(target):
        raise RuntimeError(f"model archive did not produce {target}/config.ini")
    return target


def _stub_df_io() -> None:
    mod = types.ModuleType("df.io")

    def unavailable(*_a, **_k):
        raise RuntimeError("df.io is stubbed by quran_audio; audio I/O goes through quran_audio.audio_io")

    mod.load_audio = mod.save_audio = mod.resample = unavailable
    sys.modules["df.io"] = mod


def _import_backend():
    try:
        import torch  # noqa: F401
        import libdf  # noqa: F401
    except ImportError as exc:
        raise DFNUnavailable(f"{INSTALL_HINT}\n(import failed: {exc})") from exc
    try:
        import df.io  # noqa: F401
    except Exception:
        _stub_df_io()
    try:
        from df.enhance import enhance, init_df
    except ImportError as exc:
        raise DFNUnavailable(f"{INSTALL_HINT}\n(import failed: {exc})") from exc
    return torch, init_df, enhance


class DFNDenoiser:
    def __init__(self, model_path: str | os.PathLike | None = None, threads: int | None = None) -> None:
        torch, init_df, enhance = _import_backend()
        mdir = model_dir(model_path)
        if mdir is None:
            raise DFNUnavailable("DeepFilterNet3 model not found; run `quran-audio fetch-models` "
                                 "or pass --dfn-model DIR")
        torch.set_num_threads(threads or max(1, os.cpu_count() or 1))
        self._torch = torch
        self._enhance = enhance
        self.model, self.df_state, _ = init_df(model_base_dir=str(mdir), post_filter=False, log_level="ERROR")
        self.model.eval()
        self.model_path = str(mdir)

    def _run(self, x48: np.ndarray, atten_lim_db: float | None) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            audio = torch.from_numpy(np.ascontiguousarray(x48, dtype=np.float32))[None, :]
            out = self._enhance(self.model, self.df_state, audio, pad=True, atten_lim_db=atten_lim_db)
        y = out.squeeze(0).cpu().numpy().astype(np.float64)
        return fit_length(y, len(x48))

    def enhance(self, x: np.ndarray, sr: int, atten_lim_db: float | None = None,
                chunk_s: float = 30.0, overlap_s: float = 1.0) -> np.ndarray:
        """Denoise a 1-D signal at any sample rate; returns the same rate and
        length. Long inputs are processed in overlapping chunks joined
        with raised-cosine crossfades to bound memory."""
        x = np.asarray(x, dtype=np.float64)
        x48 = resample(x, sr, DFN_SR) if sr != DFN_SR else x
        # The model collapses on very quiet input (below roughly -40 dBFS
        # speech level it treats everything as noise). Lift such input to a
        # normal level for the model and undo the gain exactly afterwards.
        gain = self.pre_gain(x48, DFN_SR)
        x48 = x48 * gain
        n = len(x48)
        chunk = int(chunk_s * DFN_SR)
        ov = int(overlap_s * DFN_SR)
        if n <= chunk + ov:
            y48 = self._run(x48, atten_lim_db)
        else:
            hop = chunk - ov
            y48 = np.zeros(n)
            wsum = np.zeros(n)
            fade = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ov) + 0.5) / ov)
            start = 0
            while start < n:
                end = min(n, start + chunk)
                seg = self._run(x48[start:end], atten_lim_db)
                w = np.ones(end - start)
                if start > 0:
                    w[:ov] = fade[: end - start]
                if end < n:
                    w[-ov:] *= fade[::-1][-(end - start):]
                y48[start:end] += seg * w
                wsum[start:end] += w
                if end >= n:
                    break
                start += hop
            y48 /= np.maximum(wsum, 1e-9)
        y48 = y48 / gain
        y = resample(y48, DFN_SR, sr) if sr != DFN_SR else y48
        return fit_length(y, len(x))

    @staticmethod
    def pre_gain(x: np.ndarray, sr: int, target_dbfs: float = -20.0) -> float:
        """Gain (>= 1) that brings the louder half of the frames to `target_dbfs`."""
        from .dynamics import frame_rms_db
        levels = frame_rms_db(x, sr)[1]
        loud = levels[levels >= np.percentile(levels, 50)]
        if loud.size == 0 or not np.isfinite(loud).any():
            return 1.0
        level = float(np.median(loud))
        return float(max(1.0, 10 ** ((target_dbfs - level) / 20.0)))


def denoise_dfn(x: np.ndarray, sr: int, atten_lim_db: float | None = None,
                model_path: str | os.PathLike | None = None,
                denoiser: DFNDenoiser | None = None) -> tuple[np.ndarray, dict]:
    den = denoiser or DFNDenoiser(model_path)
    y = den.enhance(x, sr, atten_lim_db=atten_lim_db)
    return y, {"backend": "deepfilternet3", "model_path": den.model_path,
               "atten_lim_db": atten_lim_db, "processing_sr": DFN_SR}
