"""Broadband noise reduction in the STFT domain: classical, non-generative.

Gain rule: the MMSE log-spectral-amplitude estimator (Ephraim & Malah,
1985) fused with a speech-presence probability in the "optimally
modified" form of Cohen (2002):  G = G_LSA^p * G_min^(1-p).
Noise PSD: the unbiased MMSE tracker of Gerkmann & Hendriks (2012),
initialised from the pause spectrum measured during analysis so it does
not have to learn the noise from scratch. A priori SNR: decision-directed.

Why this and not spectral subtraction: the LSA estimator minimises the
error of the *log* amplitude, which is what the ear judges, and it
produces far less "musical noise". The gain floor keeps a little natural
sounding residual noise instead of a hollow, warbling silence; above the
recording's own bandwidth edge, where there is nothing but hiss, the
floor is much deeper.

Nothing here can add content: every output coefficient is the input
coefficient times a real gain between the floor and one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.special import exp1

from .stft import STFT, frame_length_for


@dataclass
class DenoiseSettings:
    floor_db: float = -18.0          # deepest attenuation inside the usable band
    hf_floor_db: float = -40.0       # deepest attenuation above the bandwidth edge
    bandwidth_hz: float | None = None
    alpha_dd: float = 0.96           # decision-directed smoothing of the a priori SNR
    xi_min_db: float = -25.0         # a priori SNR floor
    alpha_noise: float = 0.8         # noise PSD smoothing (Gerkmann & Hendriks)
    spp_prior_snr_db: float = 15.0   # a priori SNR assumed under "speech present"
    spp_smooth_bins: int = 2         # +- bins of frequency smoothing on the fusion SPP
    profile_bounds_db: tuple[float, float] = (-6.0, 6.0)  # tracker stays this close to the pause profile
    spp_cap: float = 0.999           # Gerkmann-Hendriks "stuck" protection, made very gentle
    dd_track_bins: int = 1           # decision-directed term follows a harmonic drifting +- this many bins
    fusion: str = "soft"             # "soft" (sqrt of fixed-prior SPP), "spp" (OM-LSA), "xi" (from tracked a priori SNR), "none"
    speech_absence_prior: float = 0.5  # q in the "xi" fusion; higher = pushes uncertain bins to the floor harder
    spp_time_smooth: float = 0.0     # recursive smoothing of the fusion probability over frames (0 = off)

    def to_dict(self) -> dict:
        return asdict(self)


class SpectralDenoiser:
    """Stateful frame-by-frame processor; call it from STFT.process()."""

    def __init__(self, sr: int, n_fft: int, noise_psd: np.ndarray, settings: DenoiseSettings,
                 anchor_frames: np.ndarray | None = None, anchor_psds: np.ndarray | None = None) -> None:
        self.s = settings
        self.n_bins = n_fft // 2 + 1
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        noise_psd = np.asarray(noise_psd, dtype=np.float64)
        if len(noise_psd) != self.n_bins:
            raise ValueError("noise_psd must come from an STFT with the same frame length")
        self.profile = np.maximum(noise_psd, 1e-20)
        if anchor_frames is not None and anchor_psds is not None and len(anchor_frames) > 0:
            self.anchor_frames = np.asarray(anchor_frames, dtype=np.float64)
            self.anchor_psds = np.maximum(np.asarray(anchor_psds, dtype=np.float64), 1e-20)
        else:
            self.anchor_frames = np.zeros(1)
            self.anchor_psds = self.profile[None, :]
        self.lo = 10 ** (settings.profile_bounds_db[0] / 10)
        self.hi = 10 ** (settings.profile_bounds_db[1] / 10)
        self.noise = self._profile_at(np.zeros(1))[0]
        self.prev_s2 = np.zeros(self.n_bins)
        self.pbar = np.full(self.n_bins, 0.5)
        self.p_prev = np.zeros(self.n_bins)
        self.xi_h1 = 10 ** (settings.spp_prior_snr_db / 10)
        self.xi_min = 10 ** (settings.xi_min_db / 10)
        self.floor = self._gain_floor(freqs)
        k = max(0, int(settings.spp_smooth_bins))
        kernel = np.hanning(2 * k + 3)[1:-1]
        self.kernel = kernel / kernel.sum()
        self.frames = 0

    def _gain_floor(self, freqs: np.ndarray) -> np.ndarray:
        s = self.s
        floor_db = np.full(len(freqs), s.floor_db)
        if s.bandwidth_hz is not None and s.bandwidth_hz < freqs[-1]:
            edge = s.bandwidth_hz * 1.05
            edge_hi = edge * 2 ** (1.0 / 3)
            lf = np.log(np.maximum(freqs, 1.0))
            ramp = np.clip((lf - np.log(edge)) / (np.log(edge_hi) - np.log(edge)), 0.0, 1.0)
            floor_db = s.floor_db + ramp * (s.hf_floor_db - s.floor_db)
        return 10 ** (floor_db / 20)

    def _profile_at(self, frames: np.ndarray) -> np.ndarray:
        """Noise profile for each frame index: linear interpolation between
        the pause anchors, held constant beyond the first and last."""
        af, ap = self.anchor_frames, self.anchor_psds
        if len(af) == 1:
            return np.repeat(ap, len(frames), axis=0)
        i = np.clip(np.searchsorted(af, frames), 1, len(af) - 1)
        w = np.clip((frames - af[i - 1]) / np.maximum(af[i] - af[i - 1], 1e-9), 0.0, 1.0)
        return ap[i - 1] * (1.0 - w)[:, None] + ap[i] * w[:, None]

    def __call__(self, spec: np.ndarray, m0: int) -> np.ndarray:
        out = np.empty_like(spec)
        prof = self._profile_at(m0 + np.arange(spec.shape[0], dtype=np.float64))
        noise, prev_s2, pbar = self.noise, self.prev_s2, self.pbar
        lo, hi, cap = self.lo, self.hi, self.s.spp_cap
        a_n, a_dd = self.s.alpha_noise, self.s.alpha_dd
        xi_h1 = self.xi_h1
        c_h1 = xi_h1 / (1.0 + xi_h1)
        floor, kernel, xi_min = self.floor, self.kernel, self.xi_min
        track = max(0, int(self.s.dd_track_bins))
        fusion = self.s.fusion
        q = self.s.speech_absence_prior
        q_ratio = q / max(1.0 - q, 1e-6)
        a_p = float(np.clip(self.s.spp_time_smooth, 0.0, 0.99))
        p_prev = self.p_prev
        for i in range(spec.shape[0]):
            y = spec[i]
            y2 = y.real * y.real + y.imag * y.imag
            gamma = y2 / noise
            # speech presence probability with a fixed prior (P(H1) = 0.5)
            spp = 1.0 / (1.0 + (1.0 + xi_h1) * np.exp(-np.minimum(gamma * c_h1, 700.0)))
            pbar = 0.9 * pbar + 0.1 * spp
            spp = np.where(pbar > 0.99, np.minimum(spp, cap), spp)
            # a priori SNR, decision-directed; the max over neighbouring bins
            # keeps a harmonic that drifted a bin (vibrato, melodic pitch
            # movement) from being mistaken for a fresh onset
            prev = maximum_filter1d(prev_s2, 2 * track + 1, mode="nearest") if track else prev_s2
            xi = a_dd * prev / noise + (1.0 - a_dd) * np.maximum(gamma - 1.0, 0.0)
            xi = np.maximum(xi, xi_min)
            # MMSE-LSA gain
            v = np.maximum(xi / (1.0 + xi) * gamma, 1e-10)
            g_lsa = np.minimum(xi / (1.0 + xi) * np.exp(0.5 * exp1(v)), 1.0)
            # fuse with the (frequency-smoothed) presence probability
            if fusion == "none":
                gain = np.maximum(g_lsa, floor)
            else:
                if fusion == "xi":
                    # Cohen (2002): presence given the tracked a priori SNR
                    p = 1.0 / (1.0 + q_ratio * (1.0 + xi) * np.exp(-np.minimum(v, 700.0)))
                else:
                    p = spp
                p = np.convolve(p, kernel, mode="same")
                if a_p > 0.0:
                    p = a_p * p_prev + (1.0 - a_p) * p
                    p_prev = p
                if fusion == "soft":
                    p = np.sqrt(p)
                gain = np.maximum(g_lsa ** p * floor ** (1.0 - p), floor)
            s_hat = gain * y
            prev_s2 = s_hat.real * s_hat.real + s_hat.imag * s_hat.imag
            # noise update after the gain so this frame's gamma used the old estimate
            noise = a_n * noise + (1.0 - a_n) * ((1.0 - spp) * y2 + spp * noise)
            noise = np.clip(noise, lo * prof[i], hi * prof[i])
            out[i] = s_hat
        self.noise, self.prev_s2, self.pbar, self.p_prev = noise, prev_s2, pbar, p_prev
        self.frames += spec.shape[0]
        return out


def noise_profile(x: np.ndarray, st: STFT, quantile: float = 0.2) -> np.ndarray:
    """Mean power spectrum of the quietest `quantile` of frames. Used when
    no analysis-derived pause spectrum is supplied."""
    energies = []
    for _, power in st.power_blocks(x):
        energies.append(power.mean(axis=1))
    e = np.concatenate(energies)
    thresh = np.quantile(e, quantile)
    acc = np.zeros(st.n_bins)
    count = 0
    for m0, power in st.power_blocks(x):
        sel = e[m0:m0 + power.shape[0]] <= thresh
        if sel.any():
            acc += power[sel].sum(axis=0)
            count += int(sel.sum())
    return acc / max(count, 1)


def denoise(x: np.ndarray, sr: int, analysis=None, settings: DenoiseSettings | None = None,
            block_frames: int = 1024) -> tuple[np.ndarray, dict]:
    """Return (enhanced, info). `analysis` is the `Analysis` of `x` (same
    sample rate); it supplies the pause-anchored noise profile and the
    bandwidth edge. Without it a profile is taken from the quietest frames."""
    settings = settings or DenoiseSettings()
    n_fft = frame_length_for(sr)
    st = STFT(n_fft)
    anchors = (None, None)
    if analysis is not None and analysis.noise_psd is not None and analysis.n_fft == n_fft:
        noise_psd = analysis.noise_psd
        anchors = (analysis.pause_frames, analysis.pause_psds)
        if settings.bandwidth_hz is None:
            settings = DenoiseSettings(**{**settings.to_dict(), "bandwidth_hz": analysis.bandwidth_hz})
    else:
        noise_psd = noise_profile(x, st)
    proc = SpectralDenoiser(sr, n_fft, noise_psd, settings, *anchors)
    y = st.process(x, proc, block_frames=block_frames)
    info = {"backend": "classical-omlsa", "n_fft": n_fft, "hop": st.hop, "frames": proc.frames,
            "pause_anchors": int(len(proc.anchor_frames))}
    info.update(settings.to_dict())
    return y, info
