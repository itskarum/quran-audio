"""Optional regression on a real recording.

Set QURAN_AUDIO_REAL_FIXTURE to an audio file (a recitation transfer you
have the rights to keep locally) to run the full pipeline on its first
three minutes and check the fidelity bounds that the review established.
Skipped otherwise, so the suite stays licence-clean.
"""
import os

import pytest

from quran_audio import audio_io
from quran_audio.pipeline import enhance_signal, make_settings

FIXTURE = os.environ.get("QURAN_AUDIO_REAL_FIXTURE")


@pytest.mark.skipif(not FIXTURE, reason="QURAN_AUDIO_REAL_FIXTURE not set")
def test_real_recording_fidelity():
    a = audio_io.load(FIXTURE)
    x = a.samples[: 180 * a.sample_rate]
    res = enhance_signal(x, a.sample_rate, make_settings("standard", {"residual": True}))
    f = res.report["fidelity"]
    assert f["voice_band_retention_db"] >= -1.0, f
    assert f["projection_db"] >= -1.0, f
    if f["onset_retention_db"] is not None:
        assert f["onset_retention_db"] >= -2.0, f
    if f["tail_300ms_db"] is not None:
        assert f["tail_300ms_db"]["extra_cut"] >= -3.0, f
    assert f["in_speech_repairs"] == 0
    assert res.report["length_preserved"]
