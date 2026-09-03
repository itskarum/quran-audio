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
