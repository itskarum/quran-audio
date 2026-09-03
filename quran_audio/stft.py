"""Blocked short-time Fourier transform with exact overlap-add reconstruction.

The transform is applied block by block (a few thousand frames at a time)
so an hour-long recording never needs its whole spectrogram in memory.
Analysis and synthesis both use a periodic square-root Hann window; with a
hop that divides the frame length the squared windows sum to a constant,
so `process()` with an identity callback returns the input to within
floating-point rounding, at every sample including the edges.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class STFT:
    def __init__(self, n_fft: int, hop: int | None = None) -> None:
        hop = n_fft // 4 if hop is None else hop
        if n_fft <= 0 or hop <= 0 or n_fft % hop:
            raise ValueError("hop must divide n_fft")
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.n_bins = self.n_fft // 2 + 1
        n = np.arange(self.n_fft)
        self.window = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * n / self.n_fft))
        ratio = self.n_fft // self.hop
        wss = np.zeros(self.hop)
        for j in range(ratio):
            wss += self.window[j * self.hop:(j + 1) * self.hop] ** 2
        if not np.allclose(wss, wss[0]):
            raise ValueError("window does not satisfy the squared-COLA condition")
        self._norm = float(wss[0])
        self._pad = self.n_fft - self.hop

    # ----- geometry -------------------------------------------------------
    def n_frames(self, n_samples: int) -> int:
        return (self._pad + n_samples - 1) // self.hop + 1

    def frame_times(self, n_samples: int, sample_rate: int) -> np.ndarray:
        """Centre time (s) of every frame, relative to the original signal."""
        m = np.arange(self.n_frames(n_samples))
        return (m * self.hop + self.n_fft / 2.0 - self._pad) / sample_rate

    def freqs(self, sample_rate: int) -> np.ndarray:
        return np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)

    def _padded(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        m = self.n_frames(n)
        tail = (m - 1) * self.hop + self.n_fft - (self._pad + n)
        return np.concatenate([np.zeros(self._pad), np.asarray(x, dtype=np.float64), np.zeros(tail)])

    # ----- analysis -------------------------------------------------------
    def iter_blocks(self, x: np.ndarray, block_frames: int = 2048) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (first_frame_index, spectrum) with spectrum of shape
        (frames_in_block, n_bins), complex128."""
        xp = self._padded(x)
        total = self.n_frames(len(x))
        for m0 in range(0, total, block_frames):
            m1 = min(m0 + block_frames, total)
            seg = xp[m0 * self.hop:(m1 - 1) * self.hop + self.n_fft]
            frames = sliding_window_view(seg, self.n_fft)[::self.hop]
            yield m0, np.fft.rfft(frames * self.window, axis=1)

    def power_blocks(self, x: np.ndarray, block_frames: int = 2048) -> Iterator[tuple[int, np.ndarray]]:
        for m0, spec in self.iter_blocks(x, block_frames):
            yield m0, (spec.real ** 2 + spec.imag ** 2)

    # ----- analysis + synthesis ------------------------------------------
    def process(self, x: np.ndarray,
                block_fn: Callable[[np.ndarray, int], np.ndarray],
                block_frames: int = 2048) -> np.ndarray:
        """Run `block_fn(spectrum, first_frame_index) -> spectrum` over the
        signal and resynthesise. Output has exactly len(x) samples."""
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        xp = self._padded(x)
        yp = np.zeros_like(xp)
        total = self.n_frames(n)
        for m0 in range(0, total, block_frames):
            m1 = min(m0 + block_frames, total)
            seg = xp[m0 * self.hop:(m1 - 1) * self.hop + self.n_fft]
            frames = sliding_window_view(seg, self.n_fft)[::self.hop]
            spec = np.fft.rfft(frames * self.window, axis=1)
            out = block_fn(spec, m0)
            if out.shape != spec.shape:
                raise ValueError("block_fn must return an array of the same shape")
            frames_out = np.fft.irfft(out, n=self.n_fft, axis=1) * self.window
            for j in range(m1 - m0):
                start = (m0 + j) * self.hop
                yp[start:start + self.n_fft] += frames_out[j]
        return yp[self._pad:self._pad + n] / self._norm


    def process_many(self, xs: list[np.ndarray],
                     block_fn: Callable[[list[np.ndarray], int], list[np.ndarray]],
                     block_frames: int = 2048) -> list[np.ndarray]:
        """Like `process`, for several equal-length signals whose spectra are
        handed to `block_fn` together (so one set of gains can be applied
        to all of them). Returns one output per input."""
        xs = [np.asarray(x, dtype=np.float64) for x in xs]
        n = len(xs[0])
        if any(len(x) != n for x in xs):
            raise ValueError("all signals must have the same length")
        xps = [self._padded(x) for x in xs]
        yps = [np.zeros_like(xp) for xp in xps]
        total = self.n_frames(n)
        for m0 in range(0, total, block_frames):
            m1 = min(m0 + block_frames, total)
            specs = []
            for xp in xps:
                seg = xp[m0 * self.hop:(m1 - 1) * self.hop + self.n_fft]
                frames = sliding_window_view(seg, self.n_fft)[::self.hop]
                specs.append(np.fft.rfft(frames * self.window, axis=1))
            outs = block_fn(specs, m0)
            for yp, out in zip(yps, outs):
                frames_out = np.fft.irfft(out, n=self.n_fft, axis=1) * self.window
                for j in range(m1 - m0):
                    start = (m0 + j) * self.hop
                    yp[start:start + self.n_fft] += frames_out[j]
        return [yp[self._pad:self._pad + n] / self._norm for yp in yps]


def frame_length_for(sample_rate: int, seconds: float = 0.032, multiple: int = 64) -> int:
    """Frame length close to `seconds`, rounded to a multiple of `multiple`
    so rfft sizes stay highly composite."""
    n = int(round(sample_rate * seconds / multiple)) * multiple
    return max(n, multiple * 2)
