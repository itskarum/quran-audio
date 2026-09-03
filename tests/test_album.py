"""Album mode: shared decisions across a set of files."""
import json

import numpy as np

from conftest import at_snr, make_voice, white
from quran_audio import album, audio_io
from quran_audio.pipeline import make_settings


def test_album_shares_gain_and_tonal_decisions(tmp_path, sr):
    src = tmp_path / "in"
    src.mkdir()
    # the same voice (same crest factor) at two levels, so the input files
    # differ by 6 dB exactly; the noise scales with the voice (30 dB SNR)
    loud = make_voice(sr, 12.0, seed=1)
    quiet = 10 ** (-6 / 20) * loud
    for name, v, nseed in (("a.wav", loud, 4), ("b.wav", quiet, 5)):
        audio_io.save(src / name, v + at_snr(v, white(len(v), nseed), 30), sr, tags={"title": name})
    out = tmp_path / "out"
    summary = album.run_album([str(src / "a.wav"), str(src / "b.wav")], out, make_settings("standard"), ext=".flac", jobs=2)
    assert len(summary["files"]) == 2 and (out / "album.json").is_file()
    ra = json.loads((out / "a.report.json").read_text())
    rb = json.loads((out / "b.report.json").read_text())
    # one gain for the set, so the 6 dB difference between the files survives
    assert ra["loudness"]["gain_db"] == rb["loudness"]["gain_db"]
    la, lb = ra["loudness"]["loudness_after_lufs"], rb["loudness"]["loudness_after_lufs"]
    assert 4.5 < la - lb < 7.5, (la, lb)
    # the album as a whole sits at the target
    assert abs(summary["album_lufs"] + summary["gain_db"] - (-18.0)) < 0.5
    # the same tonal decision was applied to both
    assert ra["channels"][0]["eq"]["tonal_balance"] == "shared (album mode)" == rb["channels"][0]["eq"]["tonal_balance"]
    assert ra["album"]["files"] == 2 and summary["tonal_bands"]
    # outputs exist, carry their tags, and were limited
    ta = audio_io.load(out / "a.flac")
    assert ta.tags["title"] == "a.wav" and ta.tags["software"].startswith("quran-audio")
    assert np.max(np.abs(ta.samples)) <= 10 ** (-1.0 / 20) + 1e-3
