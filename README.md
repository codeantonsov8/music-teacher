# Music Teacher

Learn to play any music: transcribe audio into notes and sheet music, visualize
it on your instrument, play along, and get corrected.

**Status:** M0 (skeleton) and M1 (transcription lab + lean engine) are complete.
Live practice feedback and instrument visualizations come next.

## What works today

- **Transcribe audio → notes / MIDI / MusicXML** from the CLI or the local web app.
- **Lean runtime**: the vendored Basic Pitch ONNX model runs directly on ONNX
  Runtime (CUDA by default, CPU fallback) — no TensorFlow, librosa, or music21 in
  the app.
- **Research lab**: notebooks and scripts to inspect audio, look inside the model,
  compare pitch estimators, evaluate accuracy, sweep post-processing parameters,
  and benchmark GPU and live-path latency.
- **Regression gates**: synthetic fixtures with exact ground truth, golden outputs
  from the official `basic-pitch` package, and a metric baseline.

Measured on an RTX 5060 Ti: ~6 ms to transcribe 3.5 s of audio (≈600× real time),
CUDA posteriorgrams within ~1e-3 of CPU, and a YIN frame cost of ~0.06–0.16 ms
against an algorithmic latency budget of 50–100 ms.

## Quickstart

Everything Python is managed by [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # installs the gpu + dev groups (default)
uv run music-teacher serve                # opens http://127.0.0.1:8000
uv run music-teacher transcribe song.wav --out out/
```

CPU-only machines:

```bash
uv sync --no-default-groups --group cpu --group dev
```

`config/transcription.toml` controls the compute device (`auto | cpu | cuda`)
and the model's decoding thresholds.

## Project layout

```
models/                     vendored Basic Pitch ONNX model (hash-pinned, Apache-2.0)
config/transcription.toml   shared app + lab configuration
src/music_teacher/
  api/                      FastAPI app (health, device, transcribe)
  transcription/            audio I/O, lean ONNX engine, note decoding, YIN
  theory/                   minimal MIDI and MusicXML writers
  evaluation/               fixture synthesis + mir_eval metrics (dev)
  instruments/              flute/keyboard/guitar (M2+)
notebooks/                  the research lab (01-07)
scripts/                    fixtures, golden, evaluate, sweep, plot, benches
tests/                      unit, parity, and API tests
```

## The lab

```bash
uv run python scripts/make_fixtures.py      # synthetic audio + ground truth
uv run --no-project scripts/gen_golden.py   # reference outputs (isolated py3.11 env)
uv run python scripts/evaluate.py           # metrics vs baseline (exit 1 on regression)
uv run python scripts/evaluate.py --update-baseline
uv run python scripts/sweep_params.py       # threshold grid search
uv run python scripts/plot_transcription.py tests/fixtures/audio/flute_melody.wav
uv run python scripts/bench_gpu.py
uv run python scripts/bench_latency.py
uv run --group dev jupyter lab notebooks/
```

Notebooks: audio basics, model internals, estimator comparison, evaluation,
post-processing, live latency, and GPU providers. They import the same modules
the app uses, so experiments transfer directly.

## Tests

```bash
uv run pytest              # includes model inference and parity tests
uv run pytest -m "not slow"
uv run ruff check .
```

Parity tests compare the lean engine against golden outputs from
`basic-pitch==0.4.0` (generated with the ONNX artifact we vendor) and are
skipped until `scripts/gen_golden.py` has been run.

## Roadmap

- **M2** Flute "show me how to play": score/piano-roll, playback, fingering chart.
- **M3** Live practice: AudioWorklet YIN, device calibration, mistake highlighting.
- **M4** Post-session report: alignment, scoring, targeted suggestions.
- **M5** Frontend build-out.
- **M6** Keyboard, then guitar.
- **M7** Demucs stem separation, polyphonic live feedback.

## Model attribution

`models/basic_pitch_nmp.onnx` is the Basic Pitch model by Spotify AB
(Apache-2.0). See `models/README.md`, `models/LICENSE.basic-pitch`, and
`models/NOTICE.basic-pitch`.
