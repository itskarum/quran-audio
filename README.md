# quran-audio

Restoration tool for old, noisy Quran recitation recordings. It removes
tape hiss and broadband static, crackle and clicks, mains hum and rumble,
repairs clipped peaks, then balances and levels the voice, and it does all
of that **without changing what the reciter says**.

That promise is kept structurally, and then measured on every run:

- Every stage is *non-generative*. Nothing is synthesised, time-stretched,
  pitch-shifted or "re-imagined". The denoiser can only scale what is
  already in the recording (every output coefficient is the input
  coefficient times a gain between a floor and one), the click repair only
  touches a few samples at a time, and the EQ is a band-limited tilt of a
  few decibels.
- Output has exactly as many samples as the input and stays sample-aligned
  (every filter is zero-phase or delay-compensated), so the timing of
  every madd, ghunnah and qalqalah is untouched. The pipeline asserts this.
- `--residual` writes a file containing only what was removed. Play it: if
  you hear a voice in there, the preset is too strong. This is the same
  check professional restoration engineers use.
- The report carries a **fidelity block** measured on the restoration
  stages (before EQ and level): voice-band level change in speech frames,
  overall projection, onset retention, extra cut of the room decay 300 ms
  after phrases, and the number of click repairs made inside speech. It
  warns above 1 dB in the voice band, 1 dB of projection, 2 dB on onsets
  and 3 dB of extra tail cut. On the owner's recording (below) the voice
  band moves by 0.1 dB.

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

Optional, **experimental**, deep denoiser (DeepFilterNet3; CPU is fine):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build, much smaller
pip install -e '.[dfn]'
quran-audio fetch-models    # ~8 MB, verified against a pinned SHA-256
```

DeepFilterNet is opt-in (`--denoise dfn`) and stays that way: besides
noise it removes reverberation (about 9 dB of room energy on the owner's
hall recording), its level guard and 30 s chunk crossfades have been
exercised in one environment only, and it is not held to the fidelity
bounds above. The residual file shows exactly what it took.

## Use

```bash
# one file, balanced preset, plus the "what was removed" file and a JSON report
quran-audio enhance old.mp3 restored.flac --residual removed.wav --report restored.json

# measure without changing anything (noise, pauses, bandwidth, hum, transfer speed)
quran-audio analyze old.mp3

# a whole folder (recursive), FLAC out, one report per file, four files at a time
quran-audio batch ./tapes ./restored --ext flac --recursive --residuals --jobs 4

# a surah split into parts: one timbre and one level relation across the set
quran-audio batch ./surah-parts ./restored --ext flac --album
```

Presets:

| preset      | when                                                                                   |
|-------------|----------------------------------------------------------------------------------------|
| `gentle`    | Fidelity first. Lightest noise floor (-12 dB, easing to -8), small EQ, no leveler.     |
| `standard`  | Default. Floor -18 dB easing to -12 with the measured SNR, EQ capped at 6 dB, no leveler. |
| `strong`    | Heavy hiss or crackle. Dense-crackle pass, floor -28 dB easing to -16, EQ to 9 dB, leveler +-4 dB. |
| `broadcast` | Playback on small speakers or streaming: leveler +-6 dB with faster timing, -16 LUFS. This is compression. |

Every stage can be switched or tuned (`quran-audio enhance --help`):

- restoration: `--no-hum`, `--no-declick`, `--decrackle`, `--declip auto|on|off`
- denoiser: `--denoise classical|dfn|off`, `--denoise-floor DB`,
  `--no-tail-preserve`, `--dfn-atten-lim DB`, `--dfn-model DIR`
- voice: `--no-highpass`, `--no-lowpass`, `--no-tonal-balance`,
  `--tonal-strength 0..1`, `--tonal-reference recitation|speech|FILE`
- level: `--leveler` / `--no-leveler`, `--breath-db DB`, `--lufs`,
  `--no-normalize`, `--true-peak`
- transfer: `--speed-correct`, `--channels mono|keep`,
  `--mono auto|best|coherent-sum|mix|left|right`
- output: `--sr`, `--subtype PCM_16|PCM_24|FLOAT`, `--mp3-kbps`,
  `--no-dither`, `--no-provenance`

Recommended workflow for a batch of old tapes: run `standard` with
`--residuals`, listen to two or three residual files. Silence, hiss, hum
and crackle in there: good. Any trace of the voice: rerun those files with
`--preset gentle`. Read the fidelity line the tool prints; a warning there
is a reason to listen before trusting the file. Reach for `--denoise dfn`
only when the classical result still hisses and the recording is dry.

## What it does, stage by stage

1. **Decode and fold.** Any sample rate, mono or stereo. A stereo file is
   measured first: level difference, delay to +-5 ms, polarity, coherence
   in the voice band (150-3000 Hz) and each channel's noise floor. With
   coherence of 0.7 or more the channels are the same signal with their own
   noise, so they are aligned, polarity-matched and summed (3 dB less
   uncorrelated noise); below that the better channel is used alone, and a
   channel 20 dB quieter than the other is dropped. The decision and the
   measurements are in the report. `--channels keep` processes both
   channels with linked decisions: the mid decides the denoiser gains, the
   EQ and the leveler, hum and clicks are handled per channel.
2. **Analysis.** Noise floor from frames that carry signal (digital
   silence below -85 dBFS is excluded, so a muted gap cannot pose as the
   noise and a floor sitting inside the voice is caught by a comparison
   with the quietest valleys between phrases). Pauses are stretches of at
   least 0.3 s after the first half second of decay following a phrase.
   In-band SNR, usable bandwidth (the frequency above which speech no
   longer beats the noise), clipping, and the recording's own decay rate
   after phrases. A recording measuring 45 dB SNR or better skips the
   denoiser: there is nothing to remove.
3. **Hum.** The fundamental is searched in 45-65 Hz so speed-shifted
   transfers are caught, harmonics to 5 kHz, and a comb is only accepted
   when its lines also exist while the reciter is silent, so a held note
   near 100 Hz is never mistaken for hum. Each accepted line is
   demodulated at its measured frequency, the baseband is smoothed over
   one to two seconds and the reconstructed stationary sinusoid is
   subtracted: a voice harmonic sweeping through the same frequency is
   far too brief to move that estimate, and there is none of the
   half-second ringing a 1 Hz notch adds to every transient. Lines wider
   than 6 Hz (unstable hum) fall back to a zero-phase notch. The mains
   line is also a pilot tone: the report gives the transfer's speed error
   and the line width (a wow and flutter indicator); `--speed-correct`
   resamples so the line lands on 50 or 60 Hz exactly (this changes pitch,
   tempo and length, hence opt-in).
4. **Declip** (automatic when clipping is detected). Flattened peaks are
   re-synthesised by least-squares autoregressive interpolation with a
   clipping-consistency constraint.
5. **Declick / decrackle.** Short-block autoregressive modelling; samples
   whose forward *and* backward prediction errors are both outliers are
   impulsive disturbances, repaired by least-squares interpolation
   (Godsill and Rayner). Recitation is full of natural impulsive events
   (qalqalah, glottal onsets), so a candidate inside speech is repaired
   only when it is shorter than a few milliseconds *and* at least 8 dB
   above the local level; plosives are left bit-identical, and the report
   counts the repairs made inside speech.
6. **Broadband denoise.** MMSE log-spectral-amplitude gain (Ephraim and
   Malah) with the two-step a priori SNR of Plapous, Marro and Scalart so
   onsets are not lagged, combined with a speech-presence probability in
   the form G = G_LSA^p G_min^(1-p). The probability is not Cohen's OM-LSA
   estimator: it is the fixed-prior presence probability of Gerkmann and
   Hendriks (P(H1) = 0.5, 15 dB), smoothed over +-2 bins and square-rooted,
   which gates less abruptly. The noise is tracked by their unbiased MMSE
   estimator, anchored to the pause spectra (it may drift -6/+3 dB around
   them), so a long sustained vowel is never absorbed into the noise. The
   gain floor scales with the measured SNR (the preset's full floor at
   16 dB and below, its light floor at 28 dB, half of that at 40 dB), so a
   recording that is nearly clean is barely touched; above the bandwidth
   edge the floor is 40 dB. Two protections: a pitch tracker (70-400 Hz)
   marks voiced frames, and the bins on their harmonic ladder that are
   above the noise are never taken below -3 dB; and after each phrase, for
   as long as the recording's own decay rate says the room can still be
   above the noise, any bin still above it keeps the spectral-subtraction
   gain, so the decay fades into the residual instead of being gated.
7. **Voice EQ.** Rumble high-pass at 60 % of the measured low edge
   (40-80 Hz), a low-pass at 1.15 x the bandwidth edge when the recording
   is band-limited, and a tonal-balance correction measured on the
   *denoised* voice against the long-term spectrum of clean recitation
   (`--tonal-reference speech` uses the Byrne speech spectrum; a path uses
   a clean recording of the same reciter). Smoothed over one octave so
   formants are never carved, capped (6 dB at the default strength), cut
   only below 150 Hz, zero outside the usable band.
8. **Breaths** (opt-in, `--breath-db`). Inhalations between phrases are
   attenuated by up to 12 dB, never removed. A breath is a 120-900 ms
   stretch that is noise-like (no periodicity in the pitch range), tilted
   upward (centroid above 800 Hz), clearly above the floor, well below the
   speech level, and not touching a voiced frame, so a phrase-final
   fricative is not a breath. On the owner's recording, made with a
   distant microphone, no breath rises 5 dB above the floor and the option
   finds nothing to do; it is for close-miked studio recordings.
9. **Leveler and loudness.** The leveler is off in `standard` (the
   reciter's dynamics are the performance) and on in `strong` and
   `broadcast`: 2:1 above a phrase-level target, 0.6 s attack, 4 s
   release, gain frozen in pauses so noise is never pumped up. ITU-R
   BS.1770-4 loudness normalisation (K-weighting designed at the file's
   own sample rate) to -18 LUFS and a look-ahead true-peak limiter at
   -1 dBTP, measured 4x oversampled.
10. **Output.** 16-bit output gets TPDF dither, MP3 is written at a
    constant bitrate (192 kbps default), the input's tags (title, artist,
    album, track, date, genre, comment) are carried over, and a JSON
    summary of every decision is written into the comment tag so a file
    always says how it was made.

Album mode (`batch --album`) analyses the set first and shares three
decisions across it: the tonal-balance filter (designed on the set's
average voice spectrum), the leveler target, and one loudness gain for the
whole set, so the relative levels between parts survive; each file is
then limited on its own.

The JSON report records every measurement and every decision (stereo
measurements and the fold chosen, hum lines and the speed error, clicks
repaired and where, the noise floor and SNR before and after, EQ bands,
breaths, gain ranges, loudness and true peak, the fidelity block, and
every warning), so a run is fully auditable.

## Behaviour by input SNR

Measured on a synthetic recitation-like voice with white hiss, `standard`
preset, restoration stages only (no EQ or level change), against the
clean voice:

| input SNR | voice band (300-3000 Hz) in speech frames | SI-SDR after |
|----------:|------------------------------------------:|-------------:|
| 8 dB      | -1.1 dB                                   | 14.8 dB      |
| 12 dB     | -0.7 dB                                   | 18.2 dB      |
| 20 dB     | -0.2 dB                                   | 24.5 dB      |
| 30 dB     | -0.03 dB                                  | 32.1 dB      |

Below about 12 dB the denoiser starts to cost voice level, and the
fidelity warning will say so. Between 25 and 40 dB the output is never
further from the clean voice than the input was (the do-no-harm test in
the suite). Room decay after phrases is cut at most 3 dB more than the
input's own decay at 300 ms (the bound the test enforces; on the owner's
recording it measures 2.8 dB), but the input's decay already ends where the
noise begins: against a clean reference at 14 dB SNR the evaluation below
measures a larger cut.

## Measured results

`tools/evaluate.py` builds a reference and a degraded copy from a clean
recording, restores the copy with each preset and backend, and compares
with the reference. The reference is the performance as it was in the
room: the clean voice through a synthetic hall (RT60 1.2 s, decay longer
at low frequencies), breaths added in its pauses, band-limited to
4.5 kHz, with the transfer's wow (0.2 % at 0.6 Hz) and flutter (0.1 % at
7 Hz). None of that is the tool's to remove, so it counts as fidelity, not
error. The degradation is the transfer: pink and white hiss at 14 dB SNR
(or, with `--noise-file`, noise taken from the pauses of a real
transfer), a 50 Hz mains comb that shares the flutter, 400 clicks, +-3 dB
slow level drift, and an MP3 round trip at 128 kbps.

Metrics: SI-SDR (strict waveform match, dB), STOI (intelligibility, 0-1),
coherence with the reference over 150-3500 Hz (insensitive to EQ and
level, sensitive to noise and speech distortion), residual level in the
reference's pauses (dBFS), the share of injected clicks whose error fell
by 10 dB or more, hum line attenuation in the pauses (dB), voice-band
level change in speech frames against the reference (dB), extra cut of
the room decay 300 ms after phrases (dB) and the level change of the
inserted breaths (dB, 0 means kept).

Recitation (Dussary, 44.1 kHz, 60 s; the public clip is reverberant, so
it was first dried with DeepFilterNet to serve as the clean voice):

| run                | SI-SDR | STOI  | coherence | pause error | clicks fixed | hum   | voice   | tail    | breath  |
|--------------------|-------:|------:|----------:|------------:|-------------:|------:|--------:|--------:|--------:|
| degraded           |  13.0  | 0.889 |   0.832   |   -35.5     |      -       |   -   | +0.0 dB | +7.6 dB | +10.0 dB |
| classical gentle   |  18.5  | 0.902 |   0.881   |   -49.6     |    0.54      | 9.0 dB | -1.1 dB | -2.3 dB | -2.2 dB |
| classical standard |  17.9  | 0.902 |   0.881   |   -50.7     |    0.55      | 10.8 dB | -1.4 dB | -6.5 dB | -3.6 dB |
| classical strong   |  17.1  | 0.902 |   0.879   |   -51.0     |    0.60      | 12.3 dB | -1.7 dB | -9.5 dB | -4.5 dB |

The same with the hiss replaced by noise taken from the pauses of the
owner's transfer, scaled about 20 dB louder than it is in that file: hum
sidebands, MP3 artefacts and room decay, far from stationary. The tool
gains little here (STOI +0.02, 3 dB less error in the pauses) and does
little harm; it is the case the classical denoiser is weakest at, since
it can only take what stays steady through the pauses:

| run                | SI-SDR | STOI  | coherence | pause error | clicks fixed | hum   | voice   | tail     | breath  |
|--------------------|-------:|------:|----------:|------------:|-------------:|------:|--------:|---------:|--------:|
| degraded           |  12.8  | 0.838 |   0.842   |   -33.9     |      -       |   -   | +0.2 dB | +12.6 dB | +10.9 dB |
| classical gentle   |  13.2  | 0.859 |   0.869   |   -36.3     |    0.33      | 7.2 dB | -0.4 dB | +10.6 dB | +8.5 dB |
| classical standard |  12.7  | 0.859 |   0.869   |   -36.7     |    0.32      | 7.3 dB | -0.7 dB | +10.1 dB | +8.2 dB |
| classical strong   |  12.2  | 0.859 |   0.869   |   -37.0     |    0.33      | 7.5 dB | -0.9 dB | +9.5 dB  | +8.0 dB |

Clean studio speech (48 kHz, 9 s; too short for the pause, hum and tail
metrics):

| run                | SI-SDR | STOI  | coherence | pause error | clicks fixed | hum     | voice   |
|--------------------|-------:|------:|----------:|------------:|-------------:|--------:|--------:|
| degraded           |  10.1  | 0.813 |   0.791   |   -36.3     |      -       |    -    | +0.4 dB |
| classical gentle   |  19.0  | 0.898 |   0.927   |   -43.8     |    0.75      | 20.4 dB | -0.1 dB |
| classical standard |  19.0  | 0.905 |   0.934   |   -44.6     |    0.82      | 21.7 dB | -0.1 dB |
| classical strong   |  18.4  | 0.905 |   0.933   |   -44.2     |    0.83      | 21.6 dB | -0.1 dB |

How to read this: STOI and coherence are the "is the voice intact"
numbers; pause error and hum are the "is the noise gone" numbers. The
tonal-balance EQ deliberately changes the spectrum, which the voice
column and SI-SDR count as error: with `--no-eq` the `standard` preset
measures -0.8 dB in the voice band and 18.2 dB SI-SDR on the recitation,
so about 0.75 dB of the voice-band change is the denoiser's at 14 dB SNR
and the rest is EQ. The tail column is the one to watch: at 14 dB SNR
the room decay 300 ms after a phrase sits below the hiss, and what sits
below the noise goes with it (2.3 dB at `gentle`, 6.5 dB at `standard`);
the fidelity bound in the report is measured against the input's own
decay, which is all the tool can see. The inserted breaths sit 30 dB
below the speech and therefore under the hiss too, hence their loss.
Injected clicks below the hiss are not audible and are not repaired,
hence the click fractions. Rerun with:

```bash
pip install -e '.[eval]'
python tools/evaluate.py reference.wav --backends classical,dfn [--no-eq] [--noise-file transfer.mp3] [--out results/]
```

## Results on the owner's recording

A ten-minute reciter recording supplied by the project owner: 44.1 kHz
stereo MP3 transfer of a reverberant hall, 50 Hz mains comb with
harmonics, noise floor about 21 dB below the signal. `standard` preset,
measured by the report's own fidelity block (restoration stages against
the input) and by the analysis before and after:

| measure (standard preset)                          | result                                                    |
|----------------------------------------------------|-----------------------------------------------------------|
| voice band 300-3000 Hz, speech frames              | -0.09 dB                                                  |
| overall projection on the input                    | -0.03 dB                                                  |
| onsets, first frame (49 onsets)                    | -0.35 dB                                                  |
| room decay 300 ms after phrases (37 phrase ends)   | 2.8 dB below the input's own decay (bound: 3 dB; the removed hum lines counted as "decay" in the input) |
| click repairs inside speech                        | 0 (15 repairs, all in the first 0.3 s of the file)        |
| noise floor relative to the signal                 | -21.7 dB before, -31.3 dB after                           |
| in-band SNR                                        | 33.2 dB before, 42.5 dB after                             |
| pause level per 1/3-octave band                    | -4.6 to -8.7 dB                                           |
| hum                                                | 7 lines tracked (50.4, 100.8, 150.7, 250.0, 498.0, 794.3, 962.5 Hz); the 50.4 Hz mains line says the transfer runs 0.8 % fast, and the report notes that the higher lines do not fit that comb: a second hum source, each line treated at its own frequency |
| stereo                                             | left channel kept: voice-band coherence 0.18, averaging would have cancelled voice |
| tonal balance                                      | +2 to +3 dB from 160 Hz to 1.6 kHz, nothing above 2 kHz   |
| loudness                                           | -26.1 to -18.0 LUFS, true peak -1.0 dBTP, limiter took 0.28 dB at most |
| fidelity warnings                                  | none                                                      |

## Limits, stated plainly

- Lost bandwidth stays lost. A 4 kHz AM-radio transfer comes out clean but
  still 4 kHz wide; synthesising the missing octaves would mean inventing
  content, which this tool refuses to do on principle.
- Room reverb is part of the recording and the classical backend keeps
  what it can see: the part of a decay that sits below the hiss goes with
  the hiss (2.3 dB at `gentle`, 6.5 dB at `standard` 300 ms after a phrase
  at 14 dB SNR, against a clean reference). DeepFilterNet removes the room
  itself; if the room matters, stay with `--denoise classical`, and use
  `gentle` on a reverberant tape.
- Wow and flutter are measured (the hum line width) and a constant speed
  error can be corrected, but a wandering speed is not.
- Heavy distortion and dropouts longer than a few milliseconds are not
  addressed.
- Below about 12 dB input SNR the denoiser costs voice level (table
  above); `strong` lowers the residual further at a further cost, as the
  evaluation tables show. Aggressive settings can thin breathy
  consonants; the residual file will tell you.
- Hum detection needs a couple of seconds of pauses to confirm harmonics
  above 150 Hz; on very short clips only strong low harmonics are treated.
- The breath option needs breaths that stand out of the floor; on distant
  recordings it does nothing.

## Performance

Classical backend, four CPU cores: about 16 s per minute of 44.1 kHz
audio (the owner's 10.3-minute stereo file takes 164 s: analysis 29 s,
declick 24 s, denoise 37 s, hum 15 s, loudness 14 s, EQ 8 s). `batch
--jobs N` runs files in parallel. The spectral stages are processed in
blocks, but the pipeline holds several full-length copies of the signal
in memory: about 100 bytes per sample counted over all channels, so that
ten-minute stereo file peaks at 2.7 GB and an hour-long one would need
about 16 GB. Split long tapes into parts and process them with
`batch --album`, which keeps one timbre and one level relation across
the set.

## Development

```bash
pip install -e '.[test]'
pytest -q                     # 98 tests (one skipped without a real fixture), about a minute
QURAN_AUDIO_REAL_FIXTURE=/path/to/a/real/transfer.mp3 pytest -q tests/test_real.py   # optional real-audio regression
```

Layout: `quran_audio/audio_io.py` (decode, encode, resample, tags, dither),
`stft.py`, `analysis.py`, `stereo.py`, `hum.py`, `declick.py`,
`denoise.py`, `denoise_dfn.py`, `eq.py`, `dynamics.py`, `breath.py`,
`fidelity.py`, `pipeline.py` (stage order, presets, report), `album.py`,
`cli.py`. `requirements-lock.txt` pins the exact core dependency set with
hashes. DeepFilterNet (MIT/Apache-2.0, Schröter et al.) is used unmodified;
its unmaintained audio-I/O helper is stubbed out at import because the
tool does its own I/O.

## License

MIT. See `LICENSE`.
