"""Measure the live-path latency budget and engine throughput.

The live path runs in the browser; this script measures the same YIN algorithm
server-side to size the frame/window tradeoff, plus file-transcription speed.

Usage: uv run python scripts/bench_latency.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from music_teacher.evaluation.fixtures import Fixture, FixtureNote, synthesize
from music_teacher.transcription.audio import SAMPLE_RATE, load_audio
from music_teacher.transcription.monophonic import track_pitch
from music_teacher.transcription.onnx_engine import BasicPitchEngine

ROOT = Path(__file__).resolve().parent.parent


def time_yin(
    audio: np.ndarray, frame_length: int, hop: int, repeats: int = 5
) -> tuple[float, float]:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        track_pitch(audio, frame_length=frame_length, hop_length=hop)
        durations.append(time.perf_counter() - start)
    n_frames = max(len(range(0, audio.shape[0] - frame_length + 1, hop)), 1)
    per_frame = [d / n_frames for d in durations]
    return statistics.median(per_frame), statistics.quantiles(per_frame, n=20)[-1]


def main() -> int:
    # 4 seconds of an A4 flute tone
    fixture = Fixture(
        name="bench",
        notes=[FixtureNote(start=0.0, end=4.0, pitch=69)],
        timbre="flute",
        trailing_silence=0.0,
    )
    audio = synthesize(fixture)

    print("Live pitch detection (YIN, server-side reference)")
    header = f"{'window':>7} {'hop':>5} {'alg. latency':>13} {'proc/frame':>11} {'p95':>8}"
    print(header)
    print("-" * len(header))
    for frame_length, hop in ((1024, 128), (2048, 256), (4096, 256)):
        median, p95 = time_yin(audio, frame_length, hop)
        algorithmic_ms = (frame_length + hop) / SAMPLE_RATE * 1000
        print(
            f"{frame_length:>7} {hop:>5} {algorithmic_ms:>10.1f} ms "
            f"{median * 1000:>8.3f} ms {p95 * 1000:>6.3f} ms"
        )

    print("\nFrame cost is far below the algorithmic latency, so the browser")
    print("implementation is limited by the window/hop choice, not by compute.")

    engine = BasicPitchEngine(device="auto")
    engine.warmup()
    print(f"\nFile transcription (engine providers: {engine.active_providers})")
    for name in ("sine_scale", "flute_melody", "piano_chords"):
        path = ROOT / "tests" / "fixtures" / "audio" / f"{name}.wav"
        audio = load_audio(path)
        duration = audio.shape[0] / SAMPLE_RATE
        start = time.perf_counter()
        engine.transcribe(path)
        elapsed = time.perf_counter() - start
        print(f"  {name:<14} {duration:5.2f}s audio -> {elapsed * 1000:7.1f} ms "
              f"(real-time factor {elapsed / duration:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
