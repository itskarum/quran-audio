"""Breath attenuation: found where inserted, attenuated by the asked amount,
voice untouched."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from conftest import at_snr, make_voice, white
from quran_audio import breath


def test_breaths_found_and_attenuated(sr):
    v = make_voice(sr, 24.0)
    noisy = v + at_snr(v, white(len(v), 2), 45)
    rng = np.random.default_rng(9)
    sos = butter(4, [500, 4000], btype="band", fs=sr, output="sos")
    inserted = []
    for start in (4.15, 9.15, 14.15, 19.15):            # inside the 0.8 s pauses at 4, 9, 14, 19 s
        a, b = int(start * sr), int((start + 0.3) * sr)
        burst = sosfiltfilt(sos, rng.standard_normal(b - a)) * 10 ** (-32 / 20)
        burst *= np.hanning(b - a)
        noisy[a:b] += burst
        inserted.append((a, b))
    found = breath.find_breaths(noisy, sr)
    assert len(found) >= 3, found
    for a, b in inserted[:3]:
        assert any(f_a <= a + 0.1 * sr and f_b >= b - 0.1 * sr for f_a, f_b in found), (a, b, found)
    y, info = breath.attenuate_breaths(noisy, sr, -6.0)
    assert info["count"] == len(found) and info["attenuation_db"] == -6.0
    a, b = inserted[0]
    mid = slice(a + int(0.08 * sr), b - int(0.08 * sr))
    assert abs(20 * np.log10(np.std(y[mid]) / np.std(noisy[mid])) + 6.0) < 1.0
    voiced = slice(int(1.0 * sr), int(3.5 * sr))
    assert np.array_equal(y[voiced], noisy[voiced])
    y2, info2 = breath.attenuate_breaths(noisy, sr, -40.0)
    assert info2["attenuation_db"] == -12.0            # never removed, only attenuated
