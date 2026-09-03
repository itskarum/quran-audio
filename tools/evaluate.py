#!/usr/bin/env python
"""Objective evaluation: degrade a clean reference like an old recording,
restore it with each preset/backend, and measure how close the result is
to the reference.

    python tools/evaluate.py REFERENCE.wav [--snr 14] [--backends classical,dfn]
                             [--presets gentle,standard,strong] [--out DIR] [--seconds 60]

Metrics (higher is better unless noted):
  SI-SDR   scale-invariant signal-to-distortion ratio vs the band-limited reference
  STOI     short-time objective intelligibility (needs the `eval` extra), 0..1
  pause    residual level in the reference's pauses, dBFS (lower is better)
  clicks   fraction of injected clicks whose error fell by at least 10 dB
  coh      mean magnitude-squared coherence with the reference over 150-3500 Hz:
           insensitive to EQ and level, sensitive to noise and speech distortion

The degradation: 4.5 kHz band-limit, pink + white hiss at --snr, 50 Hz hum
with two harmonics, 400 clicks, +-3 dB slow level drift.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, coherence, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quran_audio import audio_io, analysis, make_settings, enhance_signal  # noqa: E402


def si_sdr(ref, est):
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    s = a * ref
    return float(10 * np.log10(np.sum(s ** 2) / (np.sum((est - s) ** 2) + 1e-20)))


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    X[1:] /= np.sqrt(np.maximum(f[1:], f[1]) / f[1])
    X[0] = 0
    y = np.fft.irfft(X, n)
    return y / np.std(y)


def degrade(clean, sr, snr_db, seed=5):
    rng = np.random.default_rng(seed)
    ref = sosfiltfilt(butter(6, 4500, fs=sr, output="sos"), clean)
    noise = 0.7 * pink(len(ref), rng) + 0.7 * rng.standard_normal(len(ref))
    noise *= np.sqrt(np.sum(ref ** 2) / np.sum(noise ** 2)) * 10 ** (-snr_db / 20)
    t = np.arange(len(ref)) / sr
    hum = 0.01 * np.sin(2 * np.pi * 50 * t) + 0.004 * np.sin(2 * np.pi * 100 * t + 1) + 0.003 * np.sin(2 * np.pi * 150 * t + 2)
    clicks = np.zeros_like(ref)
    pos = rng.integers(1000, len(ref) - 1000, 400)
    for p in pos:
        clicks[p:p + rng.integers(1, 6)] += rng.uniform(0.05, 0.4) * rng.choice([-1, 1])
    drift = 10 ** (np.sin(2 * np.pi * 0.02 * t) * 3 / 20)
    degraded = (ref + noise + hum + clicks) * drift
    scale = 0.5 / np.max(np.abs(degraded))
    return ref * scale, degraded * scale, pos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("--snr", type=float, default=14.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--start", type=float, default=5.0)
    ap.add_argument("--backends", default="classical")
    ap.add_argument("--presets", default="gentle,standard,strong")
    ap.add_argument("--out", help="directory for degraded/restored files and results.json")
    ap.add_argument("--no-eq", action="store_true", help="disable high-pass, low-pass and tonal balance")
    args = ap.parse_args()

    a = audio_io.load(args.reference)
    mono = audio_io.to_mono(a.samples, sr=a.sample_rate)[0]
    sr = a.sample_rate
    clean = mono[int(args.start * sr):int((args.start + args.seconds) * sr)]
    clean = clean - clean.mean()
    ref, degraded, click_pos = degrade(clean, sr, args.snr)
    pauses = analysis.analyze(ref, sr, hum_search=False).pause_ranges
    pz = np.concatenate([np.arange(s, e) for s, e in pauses]) if pauses else np.arange(0, sr)
    try:
        from pystoi import stoi
    except ImportError:
        stoi = None

    def band_coherence(y):
        f, c = coherence(ref, y, fs=sr, nperseg=2048)
        sel = (f >= 150) & (f <= 3500)
        return float(np.mean(c[sel]))

    def metrics(y):
        err_b = np.array([np.max(np.abs((degraded - ref)[p:p + 8])) for p in click_pos])
        err_a = np.array([np.max(np.abs((y - ref)[p:p + 8])) for p in click_pos])
        m = {"si_sdr_db": round(si_sdr(ref, y), 2), "coherence": round(band_coherence(y), 3),
             "pause_dbfs": round(float(20 * np.log10(np.std(y[pz]) + 1e-12)), 1),
             "clicks_fixed": round(float(np.mean(err_a < err_b * 10 ** (-10 / 20))), 2)}
        if stoi is not None:
            m["stoi"] = round(float(stoi(ref, y, sr)), 3)
        return m

    results = {"reference": str(args.reference), "sample_rate": sr, "seconds": len(ref) / sr, "snr_in_db": args.snr,
               "degraded": metrics(degraded), "runs": []}
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        audio_io.save(out / "reference.wav", ref, sr)
        audio_io.save(out / "degraded.wav", degraded, sr)
    print(f"reference {args.reference}: {len(ref)/sr:.0f} s at {sr} Hz, degraded at {args.snr} dB SNR")
    print(f"{'run':28s} {'SI-SDR':>7s} {'STOI':>6s} {'coh':>6s} {'pause':>7s} {'clicks':>7s} {'time':>6s}")
    d = results["degraded"]
    print(f"{'degraded':28s} {d['si_sdr_db']:7.2f} {d.get('stoi', float('nan')):6.3f} {d['coherence']:6.3f} {d['pause_dbfs']:7.1f} {d['clicks_fixed']:7.2f}")
    for backend in args.backends.split(","):
        for preset in args.presets.split(","):
            overrides = {"denoise": backend, "target_lufs": None, "leveler": False}
            if args.no_eq:
                overrides.update({"highpass": False, "lowpass": False, "tonal_balance": False})
            settings = make_settings(preset, overrides)
            t0 = time.perf_counter()
            res = enhance_signal(degraded, sr, settings)
            dt = time.perf_counter() - t0
            y = res.samples[:, 0]
            m = metrics(y)
            m.update({"backend": res.report["denoise_backend"], "preset": preset, "seconds": round(dt, 1)})
            results["runs"].append(m)
            print(f"{backend + ' ' + preset:28s} {m['si_sdr_db']:7.2f} {m.get('stoi', float('nan')):6.3f} {m['coherence']:6.3f} {m['pause_dbfs']:7.1f} {m['clicks_fixed']:7.2f} {dt:6.1f}")
            if args.out:
                audio_io.save(Path(args.out) / f"restored_{backend}_{preset}.wav", y, sr)
    if args.out:
        (Path(args.out) / "results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
