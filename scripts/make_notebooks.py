"""Generate the lab notebooks as .ipynb files.

Kept as a script so notebook content is reviewable in plain Python and stays in
sync with the package. Run: uv run python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"

SETUP = '''\
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "tests").exists():
    ROOT = ROOT.parent
AUDIO_DIR = ROOT / "tests" / "fixtures" / "audio"
GOLDEN_DIR = ROOT / "tests" / "fixtures" / "golden"
print("project root:", ROOT)
'''


def nb(cells: list[tuple[str, str]]) -> dict:
    converted = []
    for index, (kind, source) in enumerate(cells):
        if kind == "md":
            converted.append(
                {
                    "cell_type": "markdown",
                    "id": f"md-{index}",
                    "metadata": {},
                    "source": source.splitlines(keepends=True),
                }
            )
        else:
            converted.append(
                {
                    "cell_type": "code",
                    "id": f"code-{index}",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source.splitlines(keepends=True),
                }
            )
    return {
        "cells": converted,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_01() -> dict:
    return nb(
        [
            ("md", "# 01 — Audio basics\n\nWhat does the audio actually look like? Waveform, spectrogram, chroma, onsets. Everything downstream is judged against what you see here."),
            ("code", SETUP),
            ("code", '''\
from music_teacher.transcription.audio import load_audio
import librosa
import librosa.display

path = AUDIO_DIR / "flute_melody.wav"
audio = load_audio(path)
sr = 22050
print(f"{path.name}: {audio.shape[0]} samples ({audio.shape[0] / sr:.2f}s)")
'''),
            ("code", '''\
fig, ax = plt.subplots(figsize=(12, 2.5))
times = np.arange(audio.shape[0]) / sr
ax.plot(times, audio, linewidth=0.5)
ax.set(title="waveform", xlabel="time (s)", ylabel="amplitude"); ax.grid(alpha=.2)
'''),
            ("code", '''\
spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(audio.astype(float), n_fft=2048, hop_length=512)), ref=1.0)
fig, ax = plt.subplots(figsize=(12, 4))
img = librosa.display.specshow(spectrum, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=ax, cmap="magma")
fig.colorbar(img, ax=ax, format="%+2.0f dB")
'''),
            ("code", '''\
chroma = librosa.feature.chroma_cqt(y=audio.astype(float), sr=sr)
onset_env = librosa.onset.onset_strength(y=audio.astype(float), sr=sr, hop_length=512)
fig, (a, b) = plt.subplots(2, 1, figsize=(12, 5))
librosa.display.specshow(chroma, sr=sr, hop_length=512, x_axis="time", y_axis="chroma", ax=a, cmap="viridis")
a.set_title("chroma (pitch classes)")
b.plot(librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=512), onset_env)
b.set(title="onset strength", xlabel="time (s)")
fig.tight_layout()
'''),
            ("md", "**Try it:** swap in `flute_noisy.wav` or your own recording and compare. Notice which harmonics dominate — that is what confuses the transcriber."),
        ]
    )


def notebook_02() -> dict:
    return nb(
        [
            ("md", "# 02 — Basic Pitch internals\n\nInspect the model's raw output: note, onset, and contour posteriorgrams, and how thresholds change the decoded notes."),
            ("code", SETUP),
            ("code", '''\
from music_teacher.transcription.audio import load_audio
from music_teacher.transcription.onnx_engine import BasicPitchEngine

engine = BasicPitchEngine(device="auto")
engine.warmup()
print("providers:", engine.active_providers)
audio = load_audio(AUDIO_DIR / "flute_melody.wav")
output = engine.predict(audio)
for key, value in output.items():
    print(f"{key:8} {value.shape}")
'''),
            ("code", '''\
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ax, key in zip(axes, ("note", "onset", "contour")):
    ax.imshow(output[key].T, origin="lower", aspect="auto", cmap="magma")
    ax.set_ylabel(key)
axes[-1].set_xlabel("frame (~11.6 ms)")
fig.tight_layout()
'''),
            ("code", '''\
from music_teacher.transcription.notes import model_output_to_notes, filter_harmonic_duplicates

for onset_threshold in (0.3, 0.5, 0.7):
    for frame_threshold in (0.2, 0.3, 0.4):
        events = model_output_to_notes(output, onset_threshold=onset_threshold, frame_threshold=frame_threshold)
        filtered = filter_harmonic_duplicates(events)
        print(f"onset={onset_threshold} frame={frame_threshold}: raw={len(events):2d} filtered={len(filtered):2d}")
'''),
            ("code", '''\
events = filter_harmonic_duplicates(model_output_to_notes(output))
for event in events:
    print(f"{event.start:6.2f}s -> {event.end:6.2f}s  pitch={event.pitch:3d}  velocity={event.velocity:.2f}")
'''),
            ("md", "**Lab idea:** add your own post-processing functions to `notes.py`, then re-run `scripts/evaluate.py` to see whether they help."),
        ]
    )


def notebook_03() -> dict:
    return nb(
        [
            ("md", "# 03 — Estimators compared\n\npYIN (librosa) vs our YIN vs Basic Pitch on the same clip, plus per-frame timing for the live path."),
            ("code", SETUP),
            ("code", '''\
import time
from music_teacher.transcription.audio import load_audio
from music_teacher.transcription.monophonic import track_pitch, frequency_to_midi

audio = load_audio(AUDIO_DIR / "flute_melody.wav")

start = time.perf_counter()
frames = track_pitch(audio, frame_length=2048, hop_length=256)
ours_ms = (time.perf_counter() - start) * 1000
print(f"our YIN: {len(frames)} frames in {ours_ms:.1f} ms ({ours_ms / len(frames):.3f} ms/frame)")
'''),
            ("code", '''\
import librosa

start = time.perf_counter()
f0, voiced, confidence = librosa.pyin(audio.astype(float), fmin=65, fmax=2000, sr=22050, frame_length=2048, hop_length=256)
pyin_ms = (time.perf_counter() - start) * 1000
print(f"librosa pyin: {f0.shape[0]} frames in {pyin_ms:.1f} ms ({pyin_ms / f0.shape[0]:.3f} ms/frame)")
'''),
            ("code", '''\
fig, ax = plt.subplots(figsize=(12, 4))
our_times = [f.time for f in frames]
our_midi = [frequency_to_midi(f.frequency) if f.frequency else np.nan for f in frames]
ax.plot(our_times, our_midi, ".", markersize=3, label="our YIN")
pyin_times = librosa.frames_to_time(np.arange(f0.shape[0]), sr=22050, hop_length=256)
ax.plot(pyin_times, f0_to_midi := librosa.hz_to_midi(f0), ".", markersize=3, label="librosa pyin")
ax.set(title="pitch tracks", xlabel="time (s)", ylabel="MIDI pitch", ylim=(58, 84)); ax.legend(); ax.grid(alpha=.2)
'''),
            ("code", '''\
from music_teacher.transcription.onnx_engine import BasicPitchEngine
from music_teacher.transcription.notes import filter_harmonic_duplicates, model_output_to_notes

engine = BasicPitchEngine(device="auto"); engine.warmup()
start = time.perf_counter()
events = filter_harmonic_duplicates(model_output_to_notes(engine.predict(audio)))
print(f"basic pitch: {len(events)} notes in {(time.perf_counter() - start) * 1000:.1f} ms")
'''),
        ]
    )


def notebook_04() -> dict:
    return nb(
        [
            ("md", "# 04 — Evaluation\n\nMetrics against ground truth with mir_eval, plus error analysis: which notes are missed or hallucinated."),
            ("code", SETUP),
            ("code", '''\
import json
import numpy as np
from music_teacher.evaluation.fixtures import default_fixtures, write_fixture
from music_teacher.evaluation.metrics import score
from music_teacher.transcription.audio import load_audio
from music_teacher.transcription.onnx_engine import BasicPitchEngine

for fixture in default_fixtures():
    write_fixture(fixture, AUDIO_DIR)
engine = BasicPitchEngine(device="auto"); engine.warmup()
'''),
            ("code", '''\
def reference(name):
    data = json.loads((AUDIO_DIR / f"{name}.notes.json").read_text())
    intervals = np.array([[n["start"], n["end"]] for n in data["notes"]], dtype=float)
    pitches = np.array([n["pitch"] for n in data["notes"]], dtype=float)
    return intervals, pitches

rows = {}
for fixture in default_fixtures():
    ref_i, ref_p = reference(fixture.name)
    events = engine.transcribe(AUDIO_DIR / f"{fixture.name}.wav")
    est_i = np.array([[e.start, e.end] for e in events], dtype=float) if events else np.zeros((0, 2))
    est_p = np.array([e.pitch for e in events], dtype=float)
    metrics = score(ref_i, ref_p, est_i, est_p)
    rows[fixture.name] = metrics
    print(f"{fixture.name:<16} P={metrics['onset_pitch_precision']:.3f} R={metrics['onset_pitch_recall']:.3f} F1={metrics['onset_pitch_f1']:.3f}")
print("mean F1:", np.mean([m['onset_pitch_f1'] for m in rows.values()]))
'''),
            ("code", '''\
# error analysis for one fixture
name = "flute_melody"
ref_i, ref_p = reference(name)
events = engine.transcribe(AUDIO_DIR / f"{name}.wav")
est_p = [e.pitch for e in events]
print("reference:", sorted(ref_p.astype(int).tolist()))
print("estimated:", sorted(est_p))
print("extra    :", sorted(set(est_p) - set(ref_p.astype(int).tolist())))
print("missing  :", sorted(set(ref_p.astype(int).tolist()) - set(est_p)))
'''),
            ("md", "**Workflow:** change a post-processing rule, run `uv run python scripts/evaluate.py`, and only update `tests/fixtures/baseline.json` when the change is a real improvement."),
        ]
    )


def notebook_05() -> dict:
    return nb(
        [
            ("md", "# 05 — Post-processing experiments\n\nRaw model output vs harmonic filtering, velocity-ratio sweeps, and MusicXML quantization."),
            ("code", SETUP),
            ("code", '''\
from music_teacher.transcription.audio import load_audio
from music_teacher.transcription.onnx_engine import BasicPitchEngine
from music_teacher.transcription.notes import filter_harmonic_duplicates, model_output_to_notes

engine = BasicPitchEngine(device="auto"); engine.warmup()
output = engine.predict(load_audio(AUDIO_DIR / "piano_chords.wav"))
raw = model_output_to_notes(output)
print("raw notes:", len(raw))
for event in raw:
    print(f"  {event.start:5.2f}-{event.end:5.2f} p={event.pitch:3d} v={event.velocity:.2f}")
'''),
            ("code", '''\
for ratio in (0.4, 0.5, 0.65, 0.8, 0.95):
    filtered = filter_harmonic_duplicates(raw, velocity_ratio=ratio)
    print(f"velocity_ratio={ratio}: {len(raw)} -> {len(filtered)} notes")
'''),
            ("code", '''\
from music_teacher.theory import musicxml_bytes
filtered = filter_harmonic_duplicates(raw)
xml = musicxml_bytes(filtered, tempo_bpm=120, title="piano_chords")
print(xml.decode()[:600])
'''),
            ("md", "The `filter_harmonic_duplicates` rule lives in `src/music_teacher/transcription/notes.py`. Tune it here, then add a test in `tests/test_notes.py`."),
        ]
    )


def notebook_06() -> dict:
    return nb(
        [
            ("md", "# 06 — Live pitch latency\n\nSize the window/hop tradeoff for the browser AudioWorklet. Latency is the priority metric for the practice loop."),
            ("code", SETUP),
            ("code", '''\
import time, statistics
from music_teacher.evaluation.fixtures import Fixture, FixtureNote, synthesize
from music_teacher.transcription.monophonic import track_pitch, yin_pitch

audio = synthesize(Fixture(name="bench", notes=[FixtureNote(start=0.0, end=4.0, pitch=69)], timbre="flute", trailing_silence=0.0))
print(audio.shape)
'''),
            ("code", '''\
for frame_length, hop in ((1024, 128), (2048, 256), (4096, 256)):
    start = time.perf_counter()
    track_pitch(audio, frame_length=frame_length, hop_length=hop)
    elapsed = time.perf_counter() - start
    n_frames = len(range(0, audio.shape[0] - frame_length + 1, hop))
    algorithmic_ms = (frame_length + hop) / 22050 * 1000
    print(f"window={frame_length:5d} hop={hop:4d}  alg.latency={algorithmic_ms:6.1f} ms  proc/frame={elapsed / n_frames * 1000:.3f} ms")
'''),
            ("code", '''\
# simulate streaming: run YIN frame by frame and collect f0
from collections import deque

frame_length, hop = 2048, 256
f0s = []
for start in range(0, audio.shape[0] - frame_length, hop):
    frequency, confidence = yin_pitch(audio[start:start + frame_length])
    f0s.append((start / 22050, frequency, confidence))

voiced = [(t, f) for t, f, c in f0s if f > 0 and c > 0.5]
print(f"{len(voiced)}/{len(f0s)} voiced frames; median f0 = {statistics.median(f for _, f in voiced):.1f} Hz")
print(f"algorithmic latency = {(frame_length + hop) / 22050 * 1000:.1f} ms")
'''),
            ("md", "Browser target: keep the analysis window at 2048/256 at 44.1 kHz and resample-free worklet buffers; YIN itself is far below the latency budget."),
        ]
    )


def notebook_07() -> dict:
    return nb(
        [
            ("md", "# 07 — GPU providers\n\nVerify CUDA is actually used, measure throughput, and check numerical parity with CPU."),
            ("code", SETUP),
            ("code", '''\
from music_teacher.transcription.device import available_providers, describe, prepare_runtime

prepare_runtime()
print("available:", available_providers())
print(describe("auto"))
'''),
            ("code", '''\
import time
from music_teacher.transcription.audio import load_audio
from music_teacher.transcription.onnx_engine import BasicPitchEngine

audio = load_audio(AUDIO_DIR / "flute_melody.wav")
outputs, timings = {}, {}
for device in ("cpu", "cuda"):
    if device == "cuda" and "CUDAExecutionProvider" not in available_providers():
        print("CUDA unavailable"); continue
    engine = BasicPitchEngine(device=device)
    engine.warmup()
    engine.predict(audio)
    runs = []
    for _ in range(10):
        start = time.perf_counter(); engine.predict(audio); runs.append(time.perf_counter() - start)
    outputs[device] = engine.predict(audio)
    timings[device] = runs
    print(f"{device}: median {np.median(runs) * 1000:.1f} ms  min {min(runs) * 1000:.1f} ms")
'''),
            ("code", '''\
if "cpu" in outputs and "cuda" in outputs:
    for key in ("note", "onset", "contour"):
        diff = np.abs(outputs["cpu"][key] - outputs["cuda"][key])
        print(f"{key:8} max|diff| = {diff.max():.6f}  mean = {diff.mean():.8f}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, key in zip(axes, ("note", "onset", "contour")):
        ax.imshow(np.abs(outputs["cpu"][key] - outputs["cuda"][key]).T, origin="lower", aspect="auto", cmap="hot")
        ax.set_title(f"|CPU - CUDA| {key}")
    fig.tight_layout()
'''),
            ("md", "GPU is optional: `device = \"auto\"` falls back to CPU when CUDA/cuDNN is unavailable. `scripts/bench_gpu.py` is the command-line version of this notebook."),
        ]
    )


def main() -> None:
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    notebooks = {
        "01_audio_basics.ipynb": notebook_01(),
        "02_basic_pitch_internals.ipynb": notebook_02(),
        "03_estimators_compared.ipynb": notebook_03(),
        "04_evaluation.ipynb": notebook_04(),
        "05_postprocessing.ipynb": notebook_05(),
        "06_live_pitch_latency.ipynb": notebook_06(),
        "07_gpu_providers.ipynb": notebook_07(),
    }
    for name, content in notebooks.items():
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(content, indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
