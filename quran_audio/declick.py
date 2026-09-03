"""Click, crackle and clipping repair with an autoregressive signal model.

Detection: within short blocks the signal is modelled as an AR process
(autocorrelation-method LPC). Samples whose forward AND backward
prediction errors both exceed a robust threshold are impulsive
disturbances. Taking the minimum of the two errors pins the click to its
actual samples instead of the p-sample smear each predictor alone leaves.

Repair: the flagged samples are replaced by the least-squares AR
interpolation (LSAR, Godsill & Rayner) that makes the AR prediction error
over a window around the gap as small as possible, given the intact
samples on both sides.

Speech protection: recitation contains natural impulsive events (qalqalah
plosives, glottal onsets). Those disturb the residual for many
milliseconds, whereas static and record clicks are confined to a few
samples. A candidate is only repaired when the surrounding region of
elevated residual is shorter than `max_click_ms`, so plosives are left
exactly as recorded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_toeplitz, toeplitz
from scipy.signal import lfilter


@dataclass
class DeclickReport:
    clicks_repaired: int = 0
    samples_repaired: int = 0
    skipped_long: int = 0
    skipped_unstable: int = 0
    passes: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def default_order(sr: int) -> int:
    return int(min(64, max(16, round(sr / 1000) + 2)))


def ar_coefficients(seg: np.ndarray, order: int) -> np.ndarray | None:
    """Predictor a[1..p] with x[n] ~ sum_k a_k x[n-k], from a Hann-windowed
    segment. Returns None for (near) digital silence."""
    n = len(seg)
    if n <= 2 * order:
        return None
    w = np.hanning(n + 2)[1:-1]
    s = seg * w
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    r = np.fft.irfft(np.abs(np.fft.rfft(s, nfft)) ** 2, nfft)[:order + 1]
    if r[0] <= 1e-14 * n:
        return None
    r = r.copy()
    r[0] *= 1.0 + 1e-6
    try:
        a = solve_toeplitz(r[:order], r[1:order + 1])
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not mask.any():
        return np.zeros(0, int), np.zeros(0, int)
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _detect(x: np.ndarray, order: int, threshold: float, wide_threshold: float,
            block: int) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    n = len(x)
    hop = block // 2
    p = order
    mask = np.zeros(n, dtype=bool)
    wide = np.zeros(n, dtype=bool)
    coefs: dict[int, np.ndarray] = {}
    for r0 in range(0, n, hop):
        r1 = min(r0 + hop, n)
        centre = (r0 + r1) // 2
        w0 = max(0, centre - block // 2)
        w1 = min(n, centre + block // 2)
        a = ar_coefficients(x[w0:w1], p)
        if a is None:
            continue
        coefs[r0] = a
        b = np.concatenate([[1.0], -a])
        c0 = max(0, w0 - p)
        ef = lfilter(b, [1.0], x[c0:w1])[w0 - c0:]
        c1 = min(n, w1 + p)
        eb = lfilter(b, [1.0], x[w0:c1][::-1])[::-1][:w1 - w0]
        sigma = 1.4826 * float(np.median(np.abs(ef))) + 1e-12
        dmin = np.minimum(np.abs(ef), np.abs(eb))
        s0, s1 = r0 - w0, r1 - w0
        mask[r0:r1] = dmin[s0:s1] > threshold * sigma
        wide[r0:r1] = dmin[s0:s1] > wide_threshold * sigma
    mask[:p] = False
    mask[n - p:] = False
    return mask, wide, coefs


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for k in range(1, radius + 1):
        out[k:] |= mask[:-k]
        out[:-k] |= mask[k:]
    return out


def _candidate_runs(mask: np.ndarray, wide: np.ndarray, max_len: int, close_gap: int, merge_gap: int = 2):
    """Yield (start, end, ok) for each detected run after dilation and
    gap-merging; ok=False when the disturbance is too long to be a click."""
    m = _dilate(mask, 1)
    starts, ends = _runs(m)
    merged: list[list[int]] = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])
    # close small gaps in the wide mask so a decaying burst (plosive, glottal
    # onset) reads as one long disturbance rather than many short ones
    wide_closed = ~_dilate(~_dilate(wide | m, close_gap), close_gap)
    ws, we = _runs(wide_closed)
    for s, e in merged:
        j = int(np.searchsorted(ws, s, side="right")) - 1
        span = int(we[j] - ws[j]) if j >= 0 and we[j] >= e else e - s
        yield s, e, (e - s) <= max_len and span <= max_len


def lsar_fill(seg: np.ndarray, unknown: np.ndarray, a: np.ndarray) -> np.ndarray | None:
    """Least-squares AR interpolation of `seg[unknown]` given the rest.
    Returns the filled segment, or None if the problem is degenerate."""
    W = len(seg)
    p = len(a)
    if W <= p + 1 or unknown.sum() == 0:
        return None
    c = np.concatenate([-a[::-1], [1.0]])
    rows = W - p
    A = toeplitz(np.concatenate([[c[0]], np.zeros(rows - 1)]), np.concatenate([c, np.zeros(W - p - 1)]))
    known = ~unknown
    Au, Ak = A[:, unknown], A[:, known]
    rhs = -Ak @ seg[known]
    xu, *_ = np.linalg.lstsq(Au, rhs, rcond=None)
    if not np.all(np.isfinite(xu)):
        return None
    out = seg.copy()
    out[unknown] = xu
    return out


def _repair(x: np.ndarray, runs: list[tuple[int, int]], coef_at, order: int,
            context: int, report: DeclickReport) -> None:
    """Group runs whose windows overlap and solve one LSAR problem each."""
    n = len(x)
    i = 0
    while i < len(runs):
        s, e = runs[i]
        ctx = max(context, e - s)
        w0, w1 = max(0, s - ctx), min(n, e + ctx)
        j = i + 1
        while j < len(runs) and runs[j][0] - max(context, runs[j][1] - runs[j][0]) < w1:
            w1 = min(n, runs[j][1] + max(context, runs[j][1] - runs[j][0]))
            j += 1
        unknown = np.zeros(w1 - w0, dtype=bool)
        for s2, e2 in runs[i:j]:
            unknown[s2 - w0:e2 - w0] = True
        a = coef_at(s)
        filled = None
        if a is not None and unknown.sum() < 0.5 * (w1 - w0):
            filled = lsar_fill(x[w0:w1], unknown, a)
        if filled is not None:
            limit = 4.0 * (np.max(np.abs(x[w0:w1][~unknown])) + 1e-9)
            if np.max(np.abs(filled[unknown])) > limit:
                filled = None
        if filled is None:
            # fall back to linear interpolation across each run
            filled = x[w0:w1].copy()
            for s2, e2 in runs[i:j]:
                left = filled[s2 - w0 - 1] if s2 - w0 - 1 >= 0 else 0.0
                right = filled[e2 - w0] if e2 - w0 < len(filled) else 0.0
                filled[s2 - w0:e2 - w0] = np.linspace(left, right, e2 - s2 + 2)[1:-1]
            report.skipped_unstable += 1
        x[w0:w1] = filled
        for s2, e2 in runs[i:j]:
            report.clicks_repaired += 1
            report.samples_repaired += e2 - s2
        i = j


def declick(x: np.ndarray, sr: int, threshold: float = 6.0, max_click_ms: float = 2.0,
            passes: int = 2, order: int | None = None, wide_threshold: float = 2.5,
            protect: np.ndarray | None = None) -> tuple[np.ndarray, DeclickReport]:
    """Detect and repair impulsive disturbances. `threshold` is in robust
    standard deviations of the prediction error (lower = more sensitive).
    `protect` is an optional boolean mask of samples never to be touched."""
    y = np.array(x, dtype=np.float64, copy=True)
    n = len(y)
    p = order or default_order(sr)
    block = 1 << int(round(np.log2(max(0.09 * sr, 4 * p))))
    max_len = max(1, int(round(max_click_ms * 1e-3 * sr)))
    close_gap = max(4, int(0.25e-3 * sr))
    report = DeclickReport()
    if n < 4 * p:
        return y, report
    for _ in range(passes):
        report.passes += 1
        mask, wide, coefs = _detect(y, p, threshold, wide_threshold, block)
        if protect is not None:
            mask &= ~protect
        if not mask.any():
            break
        hop = block // 2
        runs = []
        for s, e, ok in _candidate_runs(mask, wide, max_len, close_gap):
            if ok:
                runs.append((s, e))
            else:
                report.skipped_long += 1
        if not runs:
            break
        _repair(y, runs, lambda s: coefs.get((s // hop) * hop), p, 2 * p, report)
    return y, report


def declip(x: np.ndarray, sr: int, threshold_ratio: float = 0.98, max_run_ms: float = 8.0,
           order: int | None = None, context_mult: int = 2) -> tuple[np.ndarray, DeclickReport]:
    """Re-synthesise flattened peaks. Runs of >= 2 samples at or above
    `threshold_ratio` x peak are treated as unknown and LSAR-interpolated
    from an AR model fitted around each run."""
    y = np.array(x, dtype=np.float64, copy=True)
    n = len(y)
    p = order or default_order(sr)
    report = DeclickReport(passes=1)
    peak = float(np.max(np.abs(y))) if n else 0.0
    if peak <= 0 or n < 8 * p:
        return y, report
    over = np.abs(y) >= threshold_ratio * peak
    starts, ends = _runs(over)
    max_len = int(max_run_ms * 1e-3 * sr)
    block = 1 << int(round(np.log2(max(0.09 * sr, 4 * p))))
    runs = []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        if e - s > max_len:
            report.skipped_long += 1
            continue
        runs.append((max(0, int(s) - 1), min(n, int(e) + 1)))

    def coef_at(s: int):
        w0 = max(0, s - block // 2)
        return ar_coefficients(y[w0:min(n, w0 + block)], p)

    # Two passes: the first AR fit sees flat tops in its context and is
    # biased; the second is estimated from the once-repaired waveform.
    for _ in range(2):
        pass_report = DeclickReport(passes=2, skipped_long=report.skipped_long)
        _repair(y, runs, coef_at, p, context_mult * p, pass_report)
        # clipping consistency: a sample that was clipped was at least as
        # large as what was recorded, so never let the repair fall below
        # the observed value, and never let it run away either
        for s, e in runs:
            seg = x[s:e]
            sign = np.sign(seg[np.argmax(np.abs(seg))])
            core = y[s:e] * sign
            y[s:e] = sign * np.clip(core, np.abs(seg), 2.5 * peak)
    return y, pass_report if runs else report
