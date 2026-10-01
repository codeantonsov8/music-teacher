"""Compare CPU and CUDA execution providers: speed and numerical parity.

Usage: uv run python scripts/bench_gpu.py [--fixture flute_melody] [--repeats 10]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from music_teacher.transcription.audio import load_audio
from music_teacher.transcription.device import available_providers
from music_teacher.transcription.onnx_engine import BasicPitchEngine

ROOT = Path(__file__).resolve().parent.parent


def bench(engine: BasicPitchEngine, audio: np.ndarray, repeats: int) -> np.ndarray:
    engine.warmup()
    engine.predict(audio)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        engine.predict(audio)
        timings.append(time.perf_counter() - start)
    return np.array(timings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="flute_melody")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    audio_path = ROOT / "tests" / "fixtures" / "audio" / f"{args.fixture}.wav"
    audio = load_audio(audio_path)
    duration = audio.shape[0] / 22_050
    print(f"fixture: {args.fixture}  duration: {duration:.2f}s  repeats: {args.repeats}")
    print(f"available providers: {available_providers()}")

    results = {}
    outputs = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and "CUDAExecutionProvider" not in available_providers():
            print("CUDA unavailable; skipping")
            continue
        engine = BasicPitchEngine(device=device)
        timings = bench(engine, audio, args.repeats)
        outputs[device] = engine.predict(audio)
        results[device] = timings
        print(
            f"{device:>4}: median {np.median(timings) * 1000:7.1f} ms  "
            f"min {timings.min() * 1000:7.1f} ms  "
            f"real-time factor {np.median(timings) / duration:.4f}"
        )

    if "cpu" in outputs and "cuda" in outputs:
        print("\nparity (max abs difference):")
        for key in ("note", "onset", "contour"):
            diff = float(np.max(np.abs(outputs["cpu"][key] - outputs["cuda"][key])))
            status = "ok" if diff < 5e-3 else "CHECK"
            print(f"  {key:<8} {diff:.6f}  {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
