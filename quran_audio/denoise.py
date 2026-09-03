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
    profile_bounds_db: tuple[float, float] = (-6.0, 3.0)  # tracker stays this close to the noise profile
    spp_cap: float = 0.999           # Gerkmann-Hendriks "stuck" protection, made very gentle
    dd_track_bins: int = 1           # decision-directed term follows a harmonic drifting +- this many bins
    fusion: str = "soft"             # "soft" (sqrt of fixed-prior SPP), "spp" (OM-LSA), "xi" (from tracked a priori SNR), "none"
    speech_absence_prior: float = 0.5  # q in the "xi" fusion; higher = pushes uncertain bins to the floor harder
    spp_time_smooth: float = 0.0     # recursive smoothing of the fusion probability over frames (0 = off)
    tail_preserve: bool = True       # relax the floor along the recording's own decay after each phrase
    two_step: bool = True            # TSNR: re-estimate the a priori SNR from this frame's first estimate (no onset lag)
    harmonic_protect: bool = True    # track f0 per frame; bins around k*f0 that are above the noise keep >= protect_db
    protect_db: float = -3.0
    pitch_lo_hz: float = 70.0
    pitch_hi_hz: float = 400.0
    pitch_max_hz: float = 5000.0     # protect harmonics up to this (and the bandwidth edge)
    voiced_contrast_db: float = 6.0  # harmonic-vs-valley contrast needed to call a frame voiced

    def to_dict(self) -> dict:
        return asdict(self)


class SpectralDenoiser:
    """Stateful frame-by-frame processor; call it from STFT.process()."""

    def __init__(self, sr: int, n_fft: int, noise_psd: np.ndarray, settings: DenoiseSettings,
                 anchor_frames: np.ndarray | None = None, anchor_psds: np.ndarray | None = None,
                 tail_allow_db: np.ndarray | None = None) -> None:
        self.s = settings
        self.tail_allow_db = tail_allow_db
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
        self.prev_gs = np.ones(self.n_bins)
        self.pbar = np.full(self.n_bins, 0.5)
        self.p_prev = np.zeros(self.n_bins)
        self.xi_h1 = 10 ** (settings.spp_prior_snr_db / 10)
        self.xi_min = 10 ** (settings.xi_min_db / 10)
        self.floor = self._gain_floor(freqs)
        k = max(0, int(settings.spp_smooth_bins))
        kernel = np.hanning(2 * k + 3)[1:-1]
        self.kernel = kernel / kernel.sum()
        self.frames = 0
        self.voiced_frames = 0
        self.f0_sum = 0.0
        self.bin_hz = sr / n_fft
        self.protect_gain = 10 ** (settings.protect_db / 20.0)
        self.protect_max_hz = min(settings.pitch_max_hz, settings.bandwidth_hz or settings.pitch_max_hz, 0.45 * sr)
        self._init_pitch()

    # ----- pitch -----------------------------------------------------------
    def _init_pitch(self) -> None:
        """Candidate fundamentals (1/48 octave apart) with the bins of their
        first harmonics and of the valleys half-way between them."""
        s = self.s
        n_oct = np.log2(s.pitch_hi_hz / s.pitch_lo_hz)
        self.cands = s.pitch_lo_hz * 2.0 ** np.arange(0.0, n_oct + 1e-9, 1.0 / 48)
        K = 10
        k = np.arange(1, K + 1)
        fh = self.cands[:, None] * k[None, :]                      # (n_cand, K)
        fv = self.cands[:, None] * (k[None, :] + 0.5)
        valid = fh < min(4000.0, (self.n_bins - 2) * self.bin_hz)
        self.pitch_valid = valid
        self.idx_lo = np.clip(np.floor(fh / self.bin_hz).astype(int), 0, self.n_bins - 1)
        self.idx_hi = np.clip(self.idx_lo + 1, 0, self.n_bins - 1)
        self.idx_v = np.clip(np.rint(fv / self.bin_hz).astype(int), 0, self.n_bins - 1)
        self.n_valid = np.maximum(valid.sum(axis=1), 1)
        self.masks = np.stack([self._harmonic_mask(f) for f in self.cands])   # (n_cand, n_bins)
        freqs = np.arange(self.n_bins) * self.bin_hz
        self.voice_band = (freqs >= 100.0) & (freqs <= min(4000.0, 0.45 * self.n_bins * 2 * self.bin_hz))

    def _pitch(self, power: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(candidate index per frame, voiced per frame) from harmonic-vs-
        valley contrast of the log power spectrum; sub- and super-octave
        candidates score low because their 'valleys' land on real harmonics.
        A frame is voiced only if it also carries energy well above the
        noise in the voice band, so pauses are never 'voiced' by chance."""
        lp = 10.0 * np.log10(power + 1e-30)
        H = np.maximum(lp[:, self.idx_lo], lp[:, self.idx_hi])      # (B, n_cand, K)
        V = lp[:, self.idx_v]
        valid = self.pitch_valid[None, :, :]
        contrast = ((H - V) * valid).sum(axis=2) / self.n_valid[None, :]
        best = np.argmax(contrast, axis=1)
        score = contrast[np.arange(contrast.shape[0]), best]
        frame_snr = power[:, self.voice_band].mean(axis=1) / max(float(noise[self.voice_band].mean()), 1e-30)
        return best, (score >= self.s.voiced_contrast_db) & (frame_snr > 10 ** (5.0 / 10))

    def _harmonic_mask(self, f0: float) -> np.ndarray:
        k = np.arange(1, int(self.protect_max_hz / f0) + 1)
        if len(k) == 0:
            return np.zeros(self.n_bins, dtype=bool)
        centres = k * f0
        half = 0.5 * self.bin_hz + 0.015 * centres
        lo = np.clip(np.ceil((centres - half) / self.bin_hz).astype(int), 0, self.n_bins - 1)
        hi = np.clip(np.floor((centres + half) / self.bin_hz).astype(int), 0, self.n_bins - 1)
        d = np.zeros(self.n_bins + 1)
        np.add.at(d, lo, 1.0)
        np.add.at(d, hi + 1, -1.0)
        return np.cumsum(d)[:self.n_bins] > 0

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
        return spec * self.gains(spec, m0)

    def gains(self, spec: np.ndarray, m0: int) -> np.ndarray:
        """Real gains in [floor, 1] for every coefficient of `spec`; the
        processor state advances as if `spec` had been processed."""
        out = np.empty(spec.shape)
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
        tail = self.tail_allow_db[m0:m0 + spec.shape[0]] if self.tail_allow_db is not None else None
        tail_kernel = np.full(5, 0.2)
        prev_gs = self.prev_gs
        base_floor = floor
        two_step = self.s.two_step
        protect = self.s.harmonic_protect
        if protect:
            power_block = spec.real * spec.real + spec.imag * spec.imag
            cand, voiced = self._pitch(power_block, noise)
            self.voiced_frames += int(voiced.sum())
            self.f0_sum += float(self.cands[cand[voiced]].sum())
        for i in range(spec.shape[0]):
            y = spec[i]
            floor = base_floor
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
            # After a phrase ends the room (or a held note) decays into the
            # noise. While the tail window is open, any bin still above the
            # noise keeps at least the spectral-subtraction gain 1 - 1/gamma
            # (from the instantaneous evidence, not the decision-directed
            # estimate, which collapses as soon as gating starts): the
            # presence fusion below cannot gate it, so the tail fades into
            # the residual noise instead of dropping off a cliff.
            # evidence that a bin is above the noise: a posteriori SNR
            # smoothed over 5 bins and recursively over ~4 frames (the bins
            # and frames of one STFT are correlated, so this is the least
            # that keeps random noise peaks from opening the floor)
            prev_gs = 0.7 * prev_gs + 0.3 * np.convolve(gamma, tail_kernel, mode="same")
            if tail is not None and i < len(tail) and tail[i]:
                floor = np.maximum(base_floor, np.where(prev_gs > 2.5, 1.0 - 1.0 / np.maximum(prev_gs, 1.0), 0.0))
            # a voiced frame keeps its harmonic ladder: the bins around k*f0
            # that are above the noise are never taken below protect_db
            if protect and voiced[i]:
                keep = self.masks[cand[i]] & (prev_gs > 2.0)
                floor = np.where(keep, np.maximum(floor, self.protect_gain), floor)
            # MMSE-LSA gain
            v = np.maximum(xi / (1.0 + xi) * gamma, 1e-10)
            g_lsa = np.minimum(xi / (1.0 + xi) * np.exp(0.5 * exp1(v)), 1.0)
            if two_step:
                # TSNR (Plapous, Marro, Scalart 2006): a second a priori SNR
                # from this frame's first estimate removes the one-frame lag
                # of the decision-directed rule at onsets
                xi2 = np.maximum(g_lsa * g_lsa * gamma, xi_min)
                v = np.maximum(xi2 / (1.0 + xi2) * gamma, 1e-10)
                g_lsa = np.minimum(xi2 / (1.0 + xi2) * np.exp(0.5 * exp1(v)), 1.0)
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
            prev_s2 = gain * gain * y2
            # noise update after the gain so this frame's gamma used the old estimate
            noise = a_n * noise + (1.0 - a_n) * ((1.0 - spp) * y2 + spp * noise)
            noise = np.clip(noise, lo * prof[i], hi * prof[i])
            out[i] = gain
        self.noise, self.prev_s2, self.pbar, self.p_prev, self.prev_gs = noise, prev_s2, pbar, p_prev, prev_gs
        self.frames += spec.shape[0]
        return out


def tail_window(n_frames: int, offset_frames, decay_db_per_s: float, floor_db: float,
                hop: int, sr: int, max_seconds: float = 3.0) -> np.ndarray | None:
    """Boolean per-frame mask: True while a phrase's decay can still be
    above the noise, i.e. from each speech offset for as long as the
    recording's own decay rate needs to fall from the speech threshold
    (10 dB over the floor) to the preset floor."""
    if offset_frames is None or len(offset_frames) == 0 or decay_db_per_s <= 0:
        return None
    seconds = min(max_seconds, (10.0 + abs(floor_db)) / decay_db_per_s)
    length = max(1, int(np.ceil(seconds * sr / hop)))
    active = np.zeros(n_frames, dtype=bool)
    for m in np.asarray(offset_frames, dtype=int):
        if 0 <= m < n_frames:
            active[m:min(n_frames, m + length)] = True
    return active


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
    st, proc, info = _processor(x, sr, analysis, settings)
    y = st.process(x, proc, block_frames=block_frames)
    info["frames"] = proc.frames
    info.update(_pitch_info(proc))
    return y, info


def denoise_linked(ref: np.ndarray, chans: list[np.ndarray], sr: int, analysis=None,
                   settings: DenoiseSettings | None = None, block_frames: int = 1024) -> tuple[list[np.ndarray], dict]:
    """Denoise several channels with one set of gains, computed from `ref`
    (normally the mid channel, whose `analysis` is given). Keeps a stereo
    image stable: no channel gets a decision the other did not."""
    st, proc, info = _processor(ref, sr, analysis, settings)

    def block(specs, m0):
        g = proc.gains(specs[0], m0)
        return [s * g for s in specs]

    outs = st.process_many([ref] + list(chans), block, block_frames=block_frames)
    info["frames"] = proc.frames
    info["linked_channels"] = len(chans)
    info.update(_pitch_info(proc))
    return outs[1:], info


def _pitch_info(proc: SpectralDenoiser) -> dict:
    if not proc.s.harmonic_protect or proc.frames == 0:
        return {}
    return {"voiced_fraction": round(proc.voiced_frames / proc.frames, 3),
            "f0_mean_hz": round(proc.f0_sum / proc.voiced_frames, 1) if proc.voiced_frames else None}


def _processor(x: np.ndarray, sr: int, analysis, settings: DenoiseSettings | None):
    settings = settings or DenoiseSettings()
    n_fft = frame_length_for(sr)
    st = STFT(n_fft)
    anchors = (None, None)
    notes: list[str] = []
    if analysis is not None and analysis.noise_psd is not None and analysis.n_fft == n_fft:
        noise_psd = analysis.noise_psd
        overrides = {}
        if settings.bandwidth_hz is None:
            overrides["bandwidth_hz"] = analysis.bandwidth_hz
        if getattr(analysis, "anchors_reliable", True):
            anchors = (analysis.pause_frames, analysis.pause_psds)
        else:
            # the valley-floor profile is calibrated to within a couple of dB;
            # a tight upper bound keeps the tracker off the analysis window's
            # leakage between harmonics, which it would otherwise learn as noise
            notes.append("pause anchors unusable; tracker running on the valley-floor profile")
        if overrides:
            settings = DenoiseSettings(**{**settings.to_dict(), **overrides})
    else:
        noise_psd = noise_profile(x, st)
    tail = None
    if settings.tail_preserve and analysis is not None and getattr(analysis, "offset_frames", None) is not None:
        tail = tail_window(st.n_frames(len(x)), analysis.offset_frames, analysis.decay_db_per_s,
                           settings.floor_db, st.hop, sr)
    proc = SpectralDenoiser(sr, n_fft, noise_psd, settings, *anchors, tail_allow_db=tail)
    info = {"backend": "classical-omlsa", "n_fft": n_fft, "hop": st.hop,
            "pause_anchors": int(len(proc.anchor_frames)) if anchors[0] is not None else 0, "notes": notes,
            "tail_preserve": tail is not None,
            "tail_window_s": round(float(min(3.0, (10.0 + abs(settings.floor_db)) / max(analysis.decay_db_per_s, 1e-9))), 2) if (tail is not None) else None,
            "decay_db_per_s": round(float(analysis.decay_db_per_s), 1) if analysis is not None else None}
    info.update(settings.to_dict())
    return st, proc, info
