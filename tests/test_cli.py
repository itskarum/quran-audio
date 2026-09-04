import json

import numpy as np
import soundfile as sf

from conftest import at_snr, make_voice, white
from quran_audio import cli


def _write(path, sr=22050, dur=4.0, seed=0):
    v = make_voice(sr, dur, seed=seed)
    sf.write(path, v + at_snr(v, white(len(v), seed + 1), 15), sr, subtype="PCM_16")
    return path


def test_enhance_and_analyze(tmp_path, capsys):
    src = _write(tmp_path / "in.wav")
    out, rep, res = tmp_path / "out.flac", tmp_path / "rep.json", tmp_path / "res.wav"
    assert cli.main(["enhance", str(src), str(out), "--report", str(rep), "--residual", str(res),
                     "--denoise", "classical", "--quiet", "--preset", "gentle"]) == 0
    assert out.is_file() and res.is_file()
    r = json.loads(rep.read_text())
    assert r["length_preserved"] and r["output"]["path"] == str(out)
    assert sf.info(str(out)).frames == sf.info(str(src)).frames
    assert cli.main(["enhance", str(src), str(out), "--quiet"]) == 2      # refuses to overwrite
    assert cli.main(["analyze", str(src), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["sample_rate"] == 22050 and "hum" in d
    assert cli.main(["analyze", str(src)]) == 0
    assert "usable band" in capsys.readouterr().out


def test_batch(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _write(src / "a.wav", seed=1)
    _write(src / "b.wav", seed=2)
    (src / "notes.txt").write_text("ignored")
    out = tmp_path / "out"
    assert cli.main(["batch", str(src), str(out), "--ext", "wav", "--quiet", "--denoise", "classical",
                     "--no-leveler", "--residuals"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["a.report.json", "a.residual.wav", "a.wav", "b.report.json", "b.residual.wav", "b.wav"]
    assert cli.main(["batch", str(tmp_path / "empty"), str(out)]) == 1


def test_bad_input_is_an_error_code(tmp_path):
    assert cli.main(["enhance", str(tmp_path / "nope.wav"), str(tmp_path / "o.wav"), "--quiet"]) == 1


def test_batch_parallel_and_album(tmp_path, voice, sr):
    from quran_audio import audio_io
    from quran_audio.cli import main
    src = tmp_path / "in"
    src.mkdir()
    audio_io.save(src / "one.wav", voice, sr)
    audio_io.save(src / "two.wav", 0.5 * voice, sr)
    assert main(["batch", str(src), str(tmp_path / "out"), "--jobs", "2", "--quiet", "--preset", "gentle", "--no-normalize"]) == 0
    assert (tmp_path / "out" / "one.wav").is_file() and (tmp_path / "out" / "two.report.json").is_file()
    assert main(["batch", str(src), str(tmp_path / "alb"), "--album", "--jobs", "2", "--quiet", "--preset", "gentle", "--ext", "flac"]) == 0
    assert (tmp_path / "alb" / "album.json").is_file() and (tmp_path / "alb" / "two.flac").is_file()


def test_breath_option_reaches_the_report(tmp_path):
    src = _write(tmp_path / "in.wav", dur=6.0)
    out, rep = tmp_path / "out.flac", tmp_path / "rep.json"
    assert cli.main(["enhance", str(src), str(out), "--report", str(rep), "--breath-db", "-6", "--quiet"]) == 0
    r = json.loads(rep.read_text())
    assert r["settings"]["breath_db"] == -6.0
    assert r["channels"][0]["breath"]["attenuation_db"] == -6.0


def test_store_false_flags_reach_settings():
    """Every --no-* / opt-out flag must be merged into Settings. A dest
    missing from the merge list is parsed and silently dropped, which is
    how --no-tail-preserve was once a no-op."""
    from quran_audio.cli import _settings_from_args, build_parser
    p = build_parser()
    args = p.parse_args(["enhance", "in.wav", "out.wav", "--no-tail-preserve",
                         "--no-hum", "--no-declick", "--no-highpass", "--no-lowpass",
                         "--no-tonal-balance"])
    s = _settings_from_args(args)
    assert s.tail_preserve is False
    assert s.hum is False and s.declick is False
    assert s.highpass is False and s.lowpass is False and s.tonal_balance is False
    # and the defaults survive when the flags are absent
    d = _settings_from_args(p.parse_args(["enhance", "in.wav", "out.wav"]))
    assert d.tail_preserve is True and d.hum is True
