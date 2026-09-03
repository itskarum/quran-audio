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
from dataclasses import dataclass, field
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


TAG_FIELDS = ("title", "artist", "album", "date", "genre", "tracknumber", "copyright", "comment", "software")

# libsndfile's MP3 encoder in constant-bitrate mode: compression_level -> kbps,
# measured with this build (44.1 kHz mono). Interpolated to honour --mp3-kbps.
_MP3_LEVEL_KBPS = [(0.0, 320.0), (0.3, 225.0), (0.6, 160.0), (0.9, 56.0)]


@dataclass
class Audio:
    """Decoded audio. `samples` is float64 with shape (n_samples, n_channels)."""

    samples: np.ndarray
    sample_rate: int
    path: str = ""
    subtype: str = ""
    tags: dict = field(default_factory=dict)

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
    tags: dict = {}
    try:
        with sf.SoundFile(str(p)) as fh:
            subtype = str(fh.subtype)
            tags = {k: getattr(fh, k) for k in TAG_FIELDS if getattr(fh, k)}
        data, sr = sf.read(str(p), dtype="float64", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        data, sr, subtype = _decode_with_ffmpeg(p, exc)
    if data.shape[0] == 0:
        raise DecodeError(f"{p}: decoded zero samples")
    if not np.all(np.isfinite(data)):
        raise DecodeError(f"{p}: decoded samples contain NaN/Inf")
    return Audio(np.ascontiguousarray(data, dtype=np.float64), int(sr), str(p), subtype, tags)


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


def mp3_compression_level(kbps: float) -> float:
    levels = np.array([lv for lv, _ in _MP3_LEVEL_KBPS])
    rates = np.array([kb for _, kb in _MP3_LEVEL_KBPS])
    return float(np.interp(-float(kbps), -rates, levels))


def save(path: str | Path, samples: np.ndarray, sample_rate: int,
         subtype: str | None = None, dither: bool = True, mp3_kbps: float | None = 192.0,
         tags: dict | None = None) -> dict[str, float | int]:
    """Encode `samples` (1-D or (n, ch)) and return {"peak", "clipped_samples", ...}.

    16-bit (and 8-bit) PCM gets TPDF dither of +-1 LSB so quiet tails
    decay into noise instead of truncation distortion. MP3 is written at a
    constant bitrate (`mp3_kbps`, default 192). `tags` (title, artist,
    album, date, comment, software, ...) go into the container's metadata
    where the format supports it (FLAC and WAV fully, MP3 most fields).
    Values outside [-1, 1] are clipped before encoding; the count is
    returned so the caller can surface it, because a limiter upstream
    should have made this impossible."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in _OUTPUT_FORMATS:
        raise EncodeError(f"{p}: unsupported output extension {ext!r}; "
                          f"use one of {sorted(_OUTPUT_FORMATS)}")
    fmt, default_subtype = _OUTPUT_FORMATS[ext]
    subtype = subtype or default_subtype
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    clipped = int(np.count_nonzero(np.abs(data) > 1.0))
    data = np.clip(data, -1.0, 1.0)
    dithered = False
    bits = {"PCM_16": 16, "PCM_S8": 8, "PCM_U8": 8}.get(subtype)
    if dither and bits:
        rng = np.random.default_rng(0)
        lsb = 2.0 ** (1 - bits)
        data = np.clip(data + (rng.random(data.shape) + rng.random(data.shape) - 1.0) * lsb, -1.0, 1.0)
        dithered = True
    p.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if fmt == "MP3" and mp3_kbps:
        kwargs = {"compression_level": mp3_compression_level(mp3_kbps), "bitrate_mode": "CONSTANT"}
    with sf.SoundFile(str(p), "w", int(sample_rate), data.shape[1], subtype=subtype, format=fmt, **kwargs) as fh:
        for key, value in (tags or {}).items():
            if key in TAG_FIELDS and value:
                try:
                    setattr(fh, key, str(value))
                except (sf.LibsndfileError, RuntimeError):
                    pass
        fh.write(data)
    return {"peak": peak, "clipped_samples": clipped, "dithered": dithered,
            "mp3_kbps": mp3_kbps if fmt == "MP3" else None}


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
