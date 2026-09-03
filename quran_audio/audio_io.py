"""Decode / encode audio and basic signal conditioning.

Only this module talks to files. Everything downstream works on 1-D
float64 arrays in [-1, 1] plus a sample rate, so the DSP stages never
have to reason about containers, codecs or channel layouts.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import firwin, kaiserord, resample_poly


class DecodeError(RuntimeError):
    """The file could not be decoded by libsndfile or ffmpeg."""


class EncodeError(RuntimeError):
    """The requested output container/encoding is not supported."""


# extension -> (libsndfile major format, default subtype)
_OUTPUT_FORMATS: dict[str, tuple[str, str]] = {
    ".wav": ("WAV", "PCM_24"),
    ".flac": ("FLAC", "PCM_24"),
    ".mp3": ("MP3", "MPEG_LAYER_III"),
    ".ogg": ("OGG", "VORBIS"),
    ".aiff": ("AIFF", "PCM_24"),
    ".aif": ("AIFF", "PCM_24"),
}


@dataclass
class Audio:
    """Decoded audio. `samples` is float64 with shape (n_samples, n_channels)."""

    samples: np.ndarray
    sample_rate: int
    path: str = ""
    subtype: str = ""

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate


def load(path: str | Path) -> Audio:
    """Decode a file. libsndfile handles WAV/FLAC/OGG/MP3/AIFF; anything
    else (M4A, WMA, video containers, ...) goes through ffmpeg if present."""
    p = Path(path)
    if not p.is_file():
        raise DecodeError(f"{p}: file not found")
    try:
        info = sf.info(str(p))
        data, sr = sf.read(str(p), dtype="float64", always_2d=True)
        subtype = str(info.subtype)
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        data, sr, subtype = _decode_with_ffmpeg(p, exc)
    if data.shape[0] == 0:
        raise DecodeError(f"{p}: decoded zero samples")
    if not np.all(np.isfinite(data)):
        raise DecodeError(f"{p}: decoded samples contain NaN/Inf")
    return Audio(np.ascontiguousarray(data, dtype=np.float64), int(sr), str(p), subtype)


def _decode_with_ffmpeg(p: Path, original_exc: Exception) -> tuple[np.ndarray, int, str]:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise DecodeError(
            f"{p}: libsndfile could not decode it ({original_exc}) and ffmpeg is "
            "not on PATH. Install ffmpeg or convert the file to WAV/FLAC first."
        ) from original_exc
    with tempfile.TemporaryDirectory(prefix="quran-audio-") as tmp:
        out = Path(tmp) / "decoded.wav"
        cmd = [exe, "-v", "error", "-nostdin", "-y", "-i", str(p),
               "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "wav", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out.is_file():
            raise DecodeError(f"{p}: ffmpeg failed: {proc.stderr.strip()[:500]}")
        data, sr = sf.read(str(out), dtype="float64", always_2d=True)
    return data, int(sr), "ffmpeg"


def save(path: str | Path, samples: np.ndarray, sample_rate: int,
         subtype: str | None = None) -> dict[str, float | int]:
    """Encode `samples` (1-D or (n, ch)) and return {"peak", "clipped_samples"}.

    Values outside [-1, 1] are clipped before PCM encoding; the count is
    returned so the caller can surface it, because a limiter upstream
    should have made this impossible."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in _OUTPUT_FORMATS:
        raise EncodeError(f"{p}: unsupported output extension {ext!r}; "
                          f"use one of {sorted(_OUTPUT_FORMATS)}")
    fmt, default_subtype = _OUTPUT_FORMATS[ext]
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    clipped = int(np.count_nonzero(np.abs(data) > 1.0))
    data = np.clip(data, -1.0, 1.0)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), data, int(sample_rate), subtype=subtype or default_subtype, format=fmt)
    return {"peak": peak, "clipped_samples": clipped}


def to_mono(samples: np.ndarray, strategy: str = "auto", sr: int | None = None):
    """Fold (n, ch) into a 1-D signal. Returns (mono, strategy_used) and,
    as a third element when a measurement was made, the StereoReport.

    Two-channel input with `auto`, `best` or `coherent-sum` goes through
    `stereo.fold`, which measures level, delay, polarity, voice-band
    coherence and per-channel noise floor before deciding. `mix`, `left`
    and `right` are literal. More than two channels are averaged."""
    if samples.ndim == 1:
        return samples, "mono", None
    if samples.shape[1] == 1:
        return samples[:, 0], "mono", None
    if strategy == "left":
        return samples[:, 0].copy(), "left", None
    if strategy == "right":
        return samples[:, 1].copy(), "right", None
    if strategy == "mix":
        return samples.mean(axis=1), "mix", None
    if strategy not in ("auto", "best", "coherent-sum"):
        raise ValueError(f"unknown mono strategy {strategy!r}")
    if samples.shape[1] > 2:
        return samples.mean(axis=1), "mix", None
    from .stereo import fold
    mono, rep = fold(samples, sr or 48000, strategy)
    return mono, rep.strategy, rep


def _resample_filter(up: int, down: int) -> np.ndarray:
    """Kaiser low-pass for polyphase resampling: 96 dB stopband, transition
    band 8 % of the lower Nyquist, cutoff centred so the stopband starts
    exactly at the lower Nyquist."""
    fc = 1.0 / max(up, down)                      # relative to the intermediate Nyquist
    width = 0.08 * fc
    numtaps, beta = kaiserord(96.0, width)
    numtaps = min(int(numtaps) | 1, 262_145)      # odd, bounded for extreme ratios
    return firwin(numtaps, fc - width / 2.0, window=("kaiser", beta))


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality polyphase resampling of a 1-D signal. Returns `x`
    itself (not a copy) when the rates already match."""
    if sr_in == sr_out:
        return x
    g = math.gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    return resample_poly(x, up, down, window=_resample_filter(up, down))


def resampled_length(n: int, sr_in: int, sr_out: int) -> int:
    if sr_in == sr_out:
        return n
    g = math.gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    return -(-n * up // down)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad the tail so that len(x) == n."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


def remove_dc(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Subtract the mean. Returns (signal, offset_removed)."""
    offset = float(np.mean(x))
    return x - offset, offset
