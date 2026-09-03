"""quran-audio: restoration of old, noisy Quran recitation recordings.

Every stage is non-generative: nothing is synthesised, time-stretched or
pitch-shifted, so the words and tajweed of the reciter are never altered.
Only noise components (hiss, static, clicks, hum, rumble) are removed and
the surviving voice is balanced and levelled.
"""

__version__ = "0.1.0"

from .pipeline import PRESETS, Result, Settings, enhance_file, enhance_signal, make_settings  # noqa: E402

__all__ = ["PRESETS", "Result", "Settings", "enhance_file", "enhance_signal", "make_settings", "__version__"]
