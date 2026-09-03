# quran-audio

Restoration tool for old, noisy Quran recitation recordings. It removes
tape hiss and broadband static, crackle and clicks, mains hum and rumble,
repairs clipped peaks, then balances and levels the voice, and it does all
of that **without ever changing what the reciter says**.

That last promise is structural, not aspirational:

- Every stage is *non-generative*. Nothing is synthesised, time-stretched,
  pitch-shifted or "re-imagined". The spectral denoisers can only scale
  what is already in the recording, the click repair only touches a few
  samples at a time, and the EQ is a gentle, band-limited tilt.
- Output has exactly as many samples as the input and stays sample-aligned
  (every filter is zero-phase or delay-compensated), so the timing of
  every madd, ghunnah and qalqalah is untouched.
- `--residual` writes a file containing only what was removed. Play it: if
  you hear a voice in there, the preset is too strong. This is the same
  check professional restoration engineers use.

## Install

Python 3.11 or newer. The core needs only numpy, scipy and libsndfile
(bundled with the `soundfile` wheel, which decodes WAV, FLAC, MP3, OGG and
AIFF). Anything else (M4A, WMA, video containers) is decoded through
`ffmpeg` if it is on your PATH.

```bash
cd quran-audio
python -m venv .venv && . .venv/bin/activate
pip install -e .            # or: pip install -r requirements-lock.txt -e .
quran-audio --version
```

Optional deep denoiser (DeepFilterNet3, runs fine on CPU, about 20x
real time on four cores):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build, much smaller
pip install -e '.[dfn]'
quran-audio fetch-models    # ~8 MB, verified against a pinned SHA-256
```

DeepFilterNet is opt-in (`--denoise dfn`), never automatic: besides noise
it removes reverberation, and on a reverberant mosque or hall recording
that strips most of the room along with the hiss and can thin sustained
notes. The residual file shows exactly what it took; if you hear the
voice in there, stay with the classical backend.

## Use

```bash
# one file, balanced preset, plus the "what was removed" file and a JSON report
quran-audio enhance old.mp3 restored.flac --residual removed.wav --report restored.json

# measure without changing anything
quran-audio analyze old.mp3

# a whole folder (recursive), FLAC out, one report per file
quran-audio batch ./tapes ./restored --ext flac --recursive --residuals
```

Presets:

| preset     | when                                                       |
|------------|------------------------------------------------------------|
| `gentle`   | Fidelity first. Lightest noise floor, no leveler, small EQ. |
| `standard` | Default. Balanced cleaning and voice preservation.         |
| `strong`   | Heavy hiss or crackle. Adds a dense-crackle pass, deeper noise floor, unlimited DeepFilterNet attenuation. |

Every stage can be switched or tuned individually (`quran-audio enhance
--help`): `--no-hum`, `--no-declick`, `--decrackle`, `--declip auto|on|off`,
`--denoise-floor DB`, `--dfn-atten-lim DB`, `--no-tonal-balance`,
`--tonal-strength`, `--no-leveler`, `--lufs`, `--no-normalize`,
`--true-peak`, `--sr`, `--channels keep`, `--mono left|right|mix`.

Recommended workflow for a batch of old tapes: run `standard` with
`--residuals`, listen to two or three residual files. Silence, hiss, hum
and crackle in there: good. Any trace of the voice: rerun those files with
`--preset gentle`. Reach for `--denoise dfn` only when the classical
result still hisses and the recording is dry.

## What it does, stage by stage

1. **Decode and condition.** Any sample rate, mono or stereo. Stereo is
   folded to mono by default because these recordings are mono at heart:
   a dead channel is dropped, an inverted channel is flipped before
   mixing, and averaging a dual-mono transfer lowers uncorrelated noise by
   3 dB for free (`--channels keep` processes channels separately).
2. **Analysis.** Noise floor, breath pauses, in-band SNR, the usable
   bandwidth (the frequency above which speech no longer beats the noise),
   clipping, and mains hum. Hum is searched anywhere in 45–65 Hz so
   speed-shifted transfers are caught; a comb is only accepted when its
   lines also exist while the reciter is silent and persist through the
   whole file, so a reciter holding a note near 100 Hz is never mistaken
   for hum.
3. **Hum removal.** Each detected harmonic is demodulated at its exact
   frequency, the baseband is smoothed over one to two seconds and the
   reconstructed stationary sinusoid is subtracted. A voice harmonic
   sweeping through the same frequency is far too brief to move that
   estimate, so it passes untouched, and there is none of the half-second
   ringing a 1 Hz notch produces at every speech transient. Only drifting
   lines fall back to a zero-phase notch. Never a blind comb at every
   multiple of 50 Hz.
4. **Declip** (automatic when clipping is detected). Flattened peaks are
   re-synthesised by least-squares autoregressive interpolation with a
   clipping-consistency constraint.
5. **Declick / decrackle.** The signal is modelled as an autoregressive
   process in short blocks; samples whose forward *and* backward prediction
   errors are both outliers are impulsive disturbances, repaired by the
   same least-squares interpolation (the classic Godsill–Rayner method).
   Recitation is full of natural impulsive events (qalqalah, glottal
   onsets), so a candidate is only repaired when the disturbance is
   shorter than a few milliseconds; plosives are left bit-identical.
6. **Broadband denoise.** Classical backend: MMSE log-spectral-amplitude
   gain (Ephraim–Malah) fused with a speech-presence probability, noise
   tracked by the unbiased MMSE estimator of Gerkmann and Hendriks and
   anchored to the spectra measured in each breath pause, so long
   sustained vowels never get absorbed into the noise estimate. A gain
   floor keeps a natural residual instead of musical noise; above the
   recording's own bandwidth edge the floor is much deeper. Deep backend:
   DeepFilterNet3, a discriminative full-band model that estimates a
   filter per time-frequency point (it cannot invent speech), with an
   optional attenuation limit.
7. **Voice EQ.** Rumble high-pass placed under the measured low edge (40–80
   Hz), a low-pass at the bandwidth edge (nothing above it but hiss), and
   a tonal-balance correction toward the long-term average speech spectrum:
   smoothed over one octave so formants are never carved, clamped to a few
   dB, cut-only below 150 Hz, zero outside the usable band.
8. **Leveler and loudness.** A slow 2:1 leveler (gain frozen in pauses so
   noise is never pumped up), ITU-R BS.1770-4 loudness normalisation to
   −18 LUFS and a look-ahead true-peak limiter at −1 dBTP.

The JSON report records every measurement and every decision (which hum
lines were notched, how many clicks were repaired, the EQ bands, gain
ranges, loudness before and after), so a run is fully auditable.

## Measured results

`tools/evaluate.py` degrades a clean reference the way an old transfer
would (4.5 kHz band-limit, pink and white hiss at 14 dB SNR, 50 Hz hum
with harmonics, 400 clicks, slow level drift), restores it with each
preset and backend, and compares with the reference. Metrics: SI-SDR
(strict waveform match, dB), STOI (intelligibility, 0–1), coherence with
the reference over 150–3500 Hz (insensitive to EQ and level, sensitive to
noise and speech distortion), residual level in the reference's pauses
(dBFS), and the share of injected clicks whose error fell by 10 dB or more.

Clean studio speech (48 kHz, 10.6 s):

| run                | SI-SDR | STOI  | coherence | pause dBFS | clicks fixed |
|--------------------|-------:|------:|----------:|-----------:|-------------:|
| degraded           |   4.8  | 0.819 |   0.621   |   −37.9    |     —        |
| classical gentle   |  10.4  | 0.875 |   0.804   |   −51.2    |    0.67      |
| classical standard |   8.2  | 0.875 |   0.815   |   −51.1    |    0.71      |
| classical strong   |   6.0  | 0.878 |   0.826   |   −50.3    |    0.75      |
| dfn gentle         |  15.3  | 0.955 |   0.926   |   −61.8    |    0.94      |
| dfn standard       |  12.2  | 0.960 |   0.929   |   −63.0    |    0.95      |
| dfn strong         |   9.3  | 0.962 |   0.932   |   −61.2    |    0.95      |

Recitation (Dussary, 44.1 kHz, 60 s; the public clip is reverberant, so
it was first dried with DeepFilterNet to serve as a reference):

| run                | SI-SDR | STOI  | coherence | pause dBFS | clicks fixed |
|--------------------|-------:|------:|----------:|-----------:|-------------:|
| degraded           |   9.8  | 0.921 |   0.846   |   −37.8    |     —        |
| classical gentle   |  11.0  | 0.923 |   0.872   |   −54.9    |    0.59      |
| classical standard |   9.3  | 0.914 |   0.856   |   −59.5    |    0.62      |
| classical strong   |   5.9  | 0.896 |   0.824   |   −65.0    |    0.60      |
| dfn gentle         |   8.9  | 0.919 |   0.872   |   −62.2    |    0.69      |
| dfn standard       |   7.7  | 0.909 |   0.867   |   −70.5    |    0.68      |
| dfn strong         |   5.2  | 0.898 |   0.866   |   −77.2    |    0.59      |

How to read this: the tonal-balance EQ deliberately changes the spectrum,
which SI-SDR counts as error (with `--no-eq` the classical standard preset
reaches 12.4 dB on the speech clip and 10.7 dB on the recitation). STOI and
coherence are the "is the voice intact" numbers; pause level is the "is
the noise gone" number. Injected clicks below the hiss are not audible and
are not repaired, hence the click fractions. Rerun with:

```bash
pip install -e '.[eval]'
python tools/evaluate.py reference.wav --backends classical,dfn [--no-eq] [--out results/]
```

## Results on a real recording

A ten-minute reciter recording supplied by the project owner: 44.1 kHz
stereo MP3 transfer of a reverberant hall, 50 Hz mains comb with
harmonics up to 900 Hz, noise floor 21 dB below the signal. Measured on
the output before EQ and level changes, against the input, in 1/3-octave
bands (the JSON report and `--residual` give the same picture per file):

| run                    | speech frames 250–2500 Hz | overall | pause bands   | hum lines 50/100/150/250 Hz |
|------------------------|---------------------------|--------:|---------------|-----------------------------|
| classical gentle       | −0.1 to −0.2 dB           | −0.10 dB | −7 to −13 dB  | −35 / −26 / −28 / −28 dB    |
| classical standard     | −0.1 to −0.3 dB           | −0.15 dB | −10 to −18 dB | −41 / −28 / −33 / −32 dB    |
| DeepFilterNet standard | −7 to −9 dB (reverb)      | −9.3 dB  | −22 to −29 dB | −54 / −48 / −47 / −47 dB    |

The classical presets leave the voice band essentially untouched and
take the hum, hiss and 24 clicks out; the residual file contains no
harmonic structure. DeepFilterNet removes the room along with the noise,
which is why it stays opt-in. On this file the tonal balance stage chose
a +2.7 dB lift at 1 kHz rising to the +3 dB clamp above 2 kHz (an old
transfer is dull against the reference spectrum); `--tonal-strength 0.3`
or `--no-tonal-balance` softens or removes that.

## Limits, stated plainly

- Lost bandwidth stays lost. A 4 kHz AM-radio transfer comes out clean but
  still 4 kHz wide; synthesising the missing octaves would mean inventing
  content, which this tool refuses to do on principle.
- Room reverb is part of the recording and the classical backend keeps it.
  DeepFilterNet removes some of it; if that matters, use `--denoise
  classical`.
- Wow and flutter, heavy distortion and dropouts longer than a few
  milliseconds are not addressed.
- The classical denoiser leaves a low, natural residual by design (the
  gain floor); `strong` lowers it at some cost in voice fidelity, as the
  tables show. Aggressive settings can thin breathy consonants; the
  residual file will tell you.
- Hum detection needs a couple of seconds of breath pauses to confirm
  harmonics above 150 Hz; on very short clips only strong low harmonics
  are notched.

## Performance

Classical backend: about 9 s per minute of 44.1 kHz mono audio on four
CPU cores, DeepFilterNet adds about 2 s per minute. Memory scales with
duration (roughly 40 bytes per sample per working copy); hour-long files
are fine on a laptop, the spectral stages are processed in blocks.

## Development

```bash
pip install -e '.[test]'
pytest -q                     # 69 tests, ~15 s, synthetic recitation-like signals
```

Layout: `quran_audio/audio_io.py` (decode/encode/resample), `stft.py`,
`analysis.py`, `hum.py`, `declick.py`, `denoise.py`, `denoise_dfn.py`,
`eq.py`, `dynamics.py`, `pipeline.py` (stage order, presets, report),
`cli.py`. `requirements-lock.txt` pins the exact core dependency set with
hashes. DeepFilterNet (MIT/Apache-2.0, Schröter et al.) is used unmodified;
its unmaintained audio-I/O helper is stubbed out at import because the
tool does its own I/O.

## License

MIT. See `LICENSE`.
