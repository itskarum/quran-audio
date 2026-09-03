"""Command-line interface: enhance, analyze, batch, fetch-models."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__, fidelity
from .pipeline import PRESETS, enhance_file, make_settings

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma",
                    ".aiff", ".aif", ".wv", ".ape", ".mp4", ".webm"}


def _log(msg: str) -> None:
    print(f"[quran-audio] {msg}", file=sys.stderr, flush=True)


def _add_processing_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("processing")
    g.add_argument("--preset", choices=sorted(PRESETS), default="standard",
                   help="gentle: light touch; standard: balanced; strong: heavy noise and crackle")
    g.add_argument("--denoise", choices=["auto", "classical", "dfn", "off"], default=None,
                   help="broadband denoiser (default classical); dfn = DeepFilterNet3, opt-in: it also removes reverb")
    g.add_argument("--denoise-floor", type=float, dest="denoise_floor_db", metavar="DB",
                   help="classical backend: deepest attenuation in the voice band, e.g. -18")
    g.add_argument("--dfn-atten-lim", type=float, dest="dfn_atten_lim_db", metavar="DB",
                   help="DeepFilterNet: cap on noise attenuation in dB (0 = unlimited)")
    g.add_argument("--dfn-model", dest="dfn_model", metavar="DIR", help="DeepFilterNet3 model directory")
    g.add_argument("--no-hum", dest="hum", action="store_false", default=None, help="skip hum detection/removal")
    g.add_argument("--no-declick", dest="declick", action="store_false", default=None, help="skip click repair")
    g.add_argument("--decrackle", dest="decrackle", action="store_true", default=None,
                   help="add a second, more sensitive pass for dense crackle (on in --preset strong)")
    g.add_argument("--declip", choices=["auto", "on", "off"], default=None, help="clipping repair (default auto)")
    g.add_argument("--no-highpass", dest="highpass", action="store_false", default=None)
    g.add_argument("--no-lowpass", dest="lowpass", action="store_false", default=None)
    g.add_argument("--no-tonal-balance", dest="tonal_balance", action="store_false", default=None)
    g.add_argument("--tonal-strength", type=float, dest="tonal_strength", metavar="0..1")
    g.add_argument("--tonal-reference", dest="tonal_reference", metavar="REF", default=None,
                   help="tonal-balance target: recitation (default), speech, or a clean recording of the reciter")
    g.add_argument("--leveler", dest="leveler", action="store_true", default=None,
                   help="phrase-level leveler (on in strong/broadcast; slow: 0.6 s attack, 4 s release, +-3 dB)")
    g.add_argument("--no-leveler", dest="leveler", action="store_false", default=None)
    g.add_argument("--speed-correct", dest="speed_correct", action="store_true", default=None,
                   help="resample so the mains line sits at 50/60 Hz exactly (changes pitch, tempo and length: opt-in)")
    g.add_argument("--no-tail-preserve", dest="tail_preserve", action="store_false", default=None,
                   help="let the denoiser cut room decay after phrases at full depth")
    g.add_argument("--lufs", type=float, dest="target_lufs", metavar="LUFS", help="loudness target (default -18)")
    g.add_argument("--no-normalize", dest="no_normalize", action="store_true", help="keep level; only limit true peaks")
    g.add_argument("--true-peak", type=float, dest="true_peak_db", metavar="DBTP", help="ceiling (default -1)")
    g.add_argument("--sr", type=int, dest="output_sr", metavar="HZ", help="output sample rate (default: input)")
    g.add_argument("--channels", choices=["mono", "keep"], default=None, help="fold to mono (default) or keep")
    g.add_argument("--mono", choices=["auto", "best", "coherent-sum", "mix", "left", "right"], dest="mono_strategy", default=None,
                   help="how to fold stereo (default auto: measured coherence decides between coherent-sum and best)")
    g.add_argument("--subtype", dest="output_subtype", metavar="SUBTYPE", help="e.g. PCM_16, PCM_24, FLOAT")
    g.add_argument("--mp3-kbps", type=int, dest="mp3_kbps", metavar="KBPS", help="constant bitrate for .mp3 output (default 192)")
    g.add_argument("--no-dither", dest="dither", action="store_false", default=None, help="no TPDF dither on 16-bit output")
    g.add_argument("--no-provenance", dest="provenance", action="store_false", default=None,
                   help="do not write the processing summary into the output's metadata")


def _settings_from_args(args: argparse.Namespace):
    keys = ["denoise", "denoise_floor_db", "dfn_atten_lim_db", "dfn_model", "hum", "declick", "decrackle",
            "declip", "highpass", "lowpass", "tonal_balance", "tonal_strength", "tonal_reference", "leveler", "target_lufs", "speed_correct",
            "dither", "mp3_kbps", "provenance",
            "true_peak_db", "output_sr", "channels", "mono_strategy", "output_subtype"]
    overrides = {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}
    if getattr(args, "no_normalize", False):
        overrides["target_lufs"] = None
    if overrides.get("dfn_atten_lim_db") == 0:
        overrides["dfn_atten_lim_db"] = None
    return make_settings(args.preset, overrides)


def cmd_enhance(args: argparse.Namespace) -> int:
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        _log(f"refusing to overwrite {out} (use --overwrite)")
        return 2
    settings = _settings_from_args(args)
    log = None if args.quiet else _log
    report = enhance_file(args.input, out, settings, residual_path=args.residual, log=log)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    if not args.quiet:
        _summary(report)
    return 0


def _summary(r: dict) -> None:
    b, a = r["analysis_before"], r["analysis_after"]
    ch = r["channels"][0]
    rel_b = b["noise_floor_dbfs"] - b["rms_dbfs"]
    rel_a = a["noise_floor_dbfs"] - a["rms_dbfs"]
    if r.get("fidelity"):
        _log("fidelity: " + fidelity.summary(r["fidelity"]))
    _log(f"backend={r['denoise_backend']} hum={len(ch.get('hum_notches', []))} notches "
         f"clicks={ch.get('declick', {}).get('clicks_repaired', 0)} "
         f"noise floor rel. to signal {rel_b:.1f} -> {rel_a:.1f} dB, "
         f"SNR {b['snr_db']:.1f} -> {a['snr_db']:.1f} dB, "
         f"loudness {r['loudness'].get('loudness_after_lufs')} LUFS, "
         f"{r['timing_s'].get('total', 0):.1f}s")


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import analyze
    from .audio_io import load, to_mono
    audio = load(args.input)
    mono, strategy, stereo_rep = to_mono(audio.samples, sr=audio.sample_rate)
    an = analyze(mono, audio.sample_rate)
    d = an.to_dict()
    d.update({"path": str(args.input), "channels": audio.n_channels, "subtype": audio.subtype, "mono_strategy": strategy,
              "stereo": stereo_rep.to_dict() if stereo_rep is not None else None})
    if args.json:
        print(json.dumps(d, indent=2))
        return 0
    print(f"{args.input}: {audio.sample_rate} Hz, {audio.n_channels} ch, {audio.duration:.1f} s, {audio.subtype}")
    print(f"  peak {d['peak_dbfs']:.1f} dBFS, rms {d['rms_dbfs']:.1f} dBFS, dc {d['dc_offset']:+.5f}, clipped runs {d['clipped_runs']}")
    print(f"  noise floor {d['noise_floor_dbfs']:.1f} dBFS, SNR ~{d['snr_db']:.1f} dB, pauses {d['pause_fraction']*100:.0f}% ({d['pause_count']})")
    print(f"  usable band {d['low_edge_hz']:.0f} - {d['bandwidth_hz']:.0f} Hz")
    hum = d["hum"]
    if hum["detected"]:
        harm = ", ".join(f"{h['freq_hz']:.1f} Hz ({h['prominence_db']:.0f} dB)" for h in hum["harmonics"])
        print(f"  hum at {hum['fundamental_hz']:.2f} Hz: {harm} [{hum['note']}]")
        if hum.get("speed_error_pct") is not None:
            print(f"  transfer speed from the mains line: {hum['speed_error_pct']:+.3f} % "
                  f"(line width {hum['line_width_hz']} Hz; a stable transfer shows ~0.4 Hz)")
    else:
        print(f"  no hum ({hum['note']})")
    return 0


def _batch_one(job: tuple) -> tuple[str, str, dict | None, str | None]:
    """One file of a batch (runs in a worker process). Returns (src, dst,
    report, error)."""
    src, dst, settings, residual = job
    try:
        report = enhance_file(src, dst, settings, residual_path=residual)
        Path(dst).with_suffix(".report.json").write_text(json.dumps(report, indent=2))
        return src, dst, report, None
    except Exception as exc:  # keep going; a batch must not die on one bad file
        return src, dst, None, f"{type(exc).__name__}: {exc}"


def cmd_batch(args: argparse.Namespace) -> int:
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    files = sorted(p for p in (in_dir.rglob("*") if args.recursive else in_dir.iterdir())
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        _log(f"no audio files found in {in_dir}")
        return 1
    settings = _settings_from_args(args)
    ext = args.ext if args.ext.startswith(".") else "." + args.ext
    jobs = max(1, int(args.jobs or 1))
    t0 = time.perf_counter()
    if args.album:
        from .album import run_album
        if args.residuals:
            _log("album mode does not write residuals; run the files singly for that")
        summary = run_album([str(f) for f in files], out_dir, settings, ext=ext, jobs=jobs, log=None if args.quiet else _log)
        warned = [f for f in summary["files"] if f["fidelity"]]
        _log(f"done: {len(summary['files'])} files, album {summary['album_lufs']} LUFS, gain {summary['gain_db']:+.2f} dB, "
             f"{len(warned)} with fidelity warnings, {time.perf_counter() - t0:.0f}s")
        return 0
    jobs_list = []
    for src in files:
        rel = src.relative_to(in_dir)
        dst = (out_dir / rel).with_suffix(ext)
        if dst.exists() and not args.overwrite:
            _log(f"skip (exists): {dst}")
            continue
        residual = dst.with_name(dst.stem + ".residual.wav") if args.residuals else None
        dst.parent.mkdir(parents=True, exist_ok=True)
        jobs_list.append((str(src), str(dst), settings, str(residual) if residual else None))
    ok = failed = 0
    if jobs == 1:
        iterator = map(_batch_one, jobs_list)
    else:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=jobs)
        iterator = pool.map(_batch_one, jobs_list)
    for i, (src, dst, report, error) in enumerate(iterator, 1):
        if error:
            failed += 1
            _log(f"[{i}/{len(jobs_list)}] FAILED {src}: {error}")
            continue
        ok += 1
        _log(f"[{i}/{len(jobs_list)}] {src} -> {dst}")
        if not args.quiet:
            _summary(report)
    if jobs > 1:
        pool.shutdown()
    _log(f"done: {ok} ok, {failed} failed, {time.perf_counter() - t0:.0f}s")
    return 0 if failed == 0 else 1


def cmd_fetch_models(args: argparse.Namespace) -> int:
    from .denoise_dfn import fetch_model
    path = fetch_model(args.dest, force=args.force)
    _log(f"DeepFilterNet3 model ready at {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quran-audio",
                                description="Restore old, noisy Quran recitation recordings without altering what is recited.")
    p.add_argument("--version", action="version", version=f"quran-audio {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("enhance", help="restore one file")
    e.add_argument("input")
    e.add_argument("output", help="output file; extension picks the format (.wav .flac .mp3 .ogg)")
    e.add_argument("--residual", metavar="PATH", help="also write what was removed (input minus denoised)")
    e.add_argument("--report", metavar="PATH", help="write a JSON report")
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--quiet", action="store_true")
    _add_processing_options(e)
    e.set_defaults(func=cmd_enhance)

    a = sub.add_parser("analyze", help="measure a file without changing it")
    a.add_argument("input")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("batch", help="restore every audio file in a directory")
    b.add_argument("input_dir")
    b.add_argument("output_dir")
    b.add_argument("--ext", default=".wav", help="output format by extension (default .wav)")
    b.add_argument("--recursive", action="store_true")
    b.add_argument("--residuals", action="store_true", help="write a .residual.wav next to each output")
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--quiet", action="store_true")
    b.add_argument("--jobs", type=int, default=1, metavar="N", help="files processed in parallel (default 1)")
    b.add_argument("--album", action="store_true",
                   help="treat the files as one set: shared tonal balance, leveler target and loudness gain")
    _add_processing_options(b)
    b.set_defaults(func=cmd_batch)

    f = sub.add_parser("fetch-models", help="download and verify the DeepFilterNet3 model")
    f.add_argument("--dest", metavar="DIR", help="cache directory (default ~/.cache/quran-audio)")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch_models)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _log("interrupted")
        return 130
    except Exception as exc:
        _log(f"error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
