"""Album mode: one set of decisions for a set of files.

A surah split into parts, or a session of several recordings of the same
reciter, should not come back with a different timbre and level per file.
Album mode analyses the whole set first and shares three decisions: the
tonal-balance filter (designed on the set's average voice spectrum), the
leveler target, and the loudness gain (one gain for the whole set, so
the reciter's relative levels between files survive), and then limits
each file on its own.
"""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import analyze, speech_spectrum
from .audio_io import load, save, to_mono
from .dynamics import frame_rms_db, integrated_loudness, limiter
from .eq import design_tonal_balance
from .pipeline import Settings, enhance_signal, provenance_tags


def _survey_one(path: str) -> dict:
    """Pass A: what this file's voice looks like, on the mono fold."""
    a = load(path)
    mono = to_mono(a.samples, "auto", a.sample_rate)[0]
    mono = mono - mono.mean()
    an = analyze(mono, a.sample_rate, hum_search=False)
    fq, sp, nz = speech_spectrum(mono, a.sample_rate)
    levels = frame_rms_db(mono, a.sample_rate)[1]
    seed = float(np.percentile(levels, 5))
    floor = float(np.median(levels[levels < seed + 6.0])) if np.any(levels < seed + 6.0) else seed
    sp_lv = levels[levels > floor + 10.0]
    return {"path": path, "sr": a.sample_rate, "freqs": fq, "clean_psd": np.maximum(sp - nz, nz),
            "low_edge_hz": an.low_edge_hz, "bandwidth_hz": an.bandwidth_hz, "duration_s": a.duration,
            "speech_level_db": float(np.median(sp_lv)) if sp_lv.size else None}


def _process_one(args: tuple) -> dict:
    """Pass B: restore with the shared decisions, no loudness normalisation
    yet; keep the result as a float WAV in the scratch directory."""
    path, tmp_path, settings = args
    a = load(path)
    res = enhance_signal(a.samples, a.sample_rate, settings)
    save(tmp_path, res.samples, res.sample_rate, subtype="FLOAT", dither=False)
    rep = res.report
    rep["input"]["path"] = path
    rep["input"]["tags"] = dict(a.tags)
    return {"path": path, "tmp": tmp_path, "sr": res.sample_rate, "report": rep,
            "loudness": float(integrated_loudness(res.samples, res.sample_rate)),
            "duration_s": res.samples.shape[0] / res.sample_rate, "tags": dict(a.tags)}


def run_album(files: list[str], out_dir: str | Path, settings: Settings, ext: str = ".flac",
              jobs: int = 1, log=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in files]
    target_lufs = settings.target_lufs

    def _log(msg: str) -> None:
        if log:
            log(msg)

    # pass A: survey
    _log(f"album: surveying {len(files)} files")
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        surveys = list(pool.map(_survey_one, files))
    ref = surveys[0]
    grid = ref["freqs"]
    psd = np.zeros_like(grid)
    total = 0.0
    for sv in surveys:
        psd += np.interp(grid, sv["freqs"], sv["clean_psd"]) * sv["duration_s"]
        total += sv["duration_s"]
    psd /= max(total, 1e-9)
    low_edge = float(np.median([sv["low_edge_hz"] for sv in surveys]))
    bandwidth = float(np.median([sv["bandwidth_hz"] for sv in surveys]))
    taps, bands = (None, [])
    if settings.tonal_balance:
        taps, bands = design_tonal_balance(psd, grid, ref["sr"], low_edge, bandwidth, settings.tonal_strength,
                                           settings.tonal_max_db, reference=settings.tonal_reference)
    lv = [sv["speech_level_db"] for sv in surveys if sv["speech_level_db"] is not None]
    leveler_target = float(np.median(lv)) if lv else None
    shared = replace(settings, tonal_taps=taps, leveler_target_db=leveler_target, target_lufs=None, residual=False)

    # pass B: restore each file with the shared decisions
    with tempfile.TemporaryDirectory(prefix="quran-audio-album-") as tmp:
        jobs_args = [(f, str(Path(tmp) / f"{i:04d}.wav"), shared) for i, f in enumerate(files)]
        with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
            results = list(pool.map(_process_one, jobs_args))
        # pass C: one gain for the set, then limit each file
        finite = [(r["loudness"], r["duration_s"]) for r in results if np.isfinite(r["loudness"])]
        if target_lufs is not None and finite:
            album_lufs = 10 * np.log10(sum(10 ** (l / 10) * d for l, d in finite) / sum(d for _, d in finite))
            gain_db = float(np.clip(target_lufs - album_lufs, -30.0, 30.0))
        else:
            album_lufs, gain_db = None, 0.0
        _log(f"album: loudness {album_lufs if album_lufs is None else round(album_lufs, 2)} LUFS, gain {gain_db:+.2f} dB for every file")
        outputs = []
        for r in results:
            y = load(r["tmp"]).samples * 10 ** (gain_db / 20.0)
            y, lim = limiter(y, r["sr"], settings.true_peak_db)
            rep = r["report"]
            rep["loudness"] = {"album_lufs": album_lufs, "gain_db": gain_db, "loudness_after_lufs": (r["loudness"] + gain_db) if np.isfinite(r["loudness"]) else None,
                               "true_peak_after_dbtp": lim["true_peak_after_dbtp"], "limiter": lim}
            rep["album"] = {"files": len(files), "tonal_bands": bands, "leveler_target_db": leveler_target}
            out_path = out_dir / (Path(r["path"]).stem + ext)
            tags = provenance_tags(rep, r["tags"]) if settings.provenance else dict(r["tags"])
            written = save(out_path, y, r["sr"], settings.output_subtype, dither=settings.dither,
                           mp3_kbps=settings.mp3_kbps, tags=tags)
            rep["output"].update({"path": str(out_path), "clipped_samples": written["clipped_samples"]})
            (out_dir / (Path(r["path"]).stem + ".report.json")).write_text(json.dumps(rep, indent=2, default=str))
            outputs.append({"input": r["path"], "output": str(out_path), "loudness_after_lufs": rep["loudness"]["loudness_after_lufs"],
                            "fidelity": rep.get("fidelity", {}).get("warnings", [])})
            _log(f"album: wrote {out_path.name}")
    summary = {"tool": f"quran-audio {__version__}", "files": outputs, "album_lufs": album_lufs, "gain_db": gain_db,
               "tonal_bands": bands, "leveler_target_db": leveler_target, "settings": settings.to_dict()}
    (out_dir / "album.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
