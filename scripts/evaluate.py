"""Evaluate transcription quality against ground-truth fixtures.

Usage:
    uv run python scripts/evaluate.py            # compare to baseline (exit 1 on regression)
    uv run python scripts/evaluate.py --update-baseline
    uv run python scripts/evaluate.py --fixture flute_melody --verbose

Metrics use mir_eval.transcription (onset+pitch, and onset+offset+pitch).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from music_teacher.config import load_config
from music_teacher.evaluation.fixtures import default_fixtures, write_fixture
from music_teacher.evaluation.metrics import score
from music_teacher.transcription.notes import NoteEvent
from music_teacher.transcription.onnx_engine import BasicPitchEngine

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "tests" / "fixtures" / "audio"
BASELINE_PATH = ROOT / "tests" / "fixtures" / "baseline.json"
REGRESSION_TOLERANCE = 0.02


def load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    intervals = np.array([[note["start"], note["end"]] for note in data["notes"]], dtype=float)
    pitches = np.array([note["pitch"] for note in data["notes"]], dtype=float)
    return intervals, pitches


def events_to_arrays(events: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    if not events:
        return np.zeros((0, 2)), np.zeros((0,))
    intervals = np.array([[note.start, note.end] for note in events], dtype=float)
    pitches = np.array([note.pitch for note in events], dtype=float)
    return intervals, pitches


def evaluate_fixture(engine: BasicPitchEngine, name: str) -> dict[str, float]:
    audio_path = AUDIO_DIR / f"{name}.wav"
    ref_intervals, ref_pitches = load_ground_truth(AUDIO_DIR / f"{name}.notes.json")
    events = engine.transcribe(audio_path)
    est_intervals, est_pitches = events_to_arrays(events)

    return {
        "reference_notes": int(ref_pitches.shape[0]),
        "estimated_notes": int(est_pitches.shape[0]),
        **score(ref_intervals, ref_pitches, est_intervals, est_pitches),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", help="evaluate a single fixture by name")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda")
    args = parser.parse_args()

    for fixture in default_fixtures():
        write_fixture(fixture, AUDIO_DIR)
    names = [args.fixture] if args.fixture else [f.name for f in default_fixtures()]
    unknown = set(names) - {f.name for f in default_fixtures()}
    if unknown:
        parser.error(f"unknown fixture(s): {', '.join(sorted(unknown))}")

    config = load_config()
    device = args.device or config.compute.device
    engine = BasicPitchEngine(config=config.basic_pitch, device=device)
    engine.warmup()

    results = {name: evaluate_fixture(engine, name) for name in names}
    header = f"{'fixture':<16} {'ref':>4} {'est':>4} {'P':>6} {'R':>6} {'F1':>6} {'F1(off)':>8}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<16} {metrics['reference_notes']:>4} {metrics['estimated_notes']:>4} "
            f"{metrics['onset_pitch_precision']:>6.3f} {metrics['onset_pitch_recall']:>6.3f} "
            f"{metrics['onset_pitch_f1']:>6.3f} {metrics['offset_pitch_f1']:>8.3f}"
        )

    mean_f1 = float(np.mean([m["onset_pitch_f1"] for m in results.values()]))
    mean_f1_offset = float(np.mean([m["offset_pitch_f1"] for m in results.values()]))
    print(f"\nmean onset+pitch F1: {mean_f1:.4f}")
    print(f"mean onset+offset+pitch F1: {mean_f1_offset:.4f}")

    if args.verbose:
        print("\nconfig:", json.dumps(config.basic_pitch.__dict__, indent=2))

    baseline = {
        "config": config.basic_pitch.__dict__,
        "device": args.device or config.compute.device,
        "fixtures": results,
        "mean_onset_pitch_f1": mean_f1,
        "mean_offset_pitch_f1": mean_f1_offset,
    }
    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"baseline updated: {BASELINE_PATH.relative_to(ROOT)}")
        return 0

    if not BASELINE_PATH.is_file():
        print("no baseline yet; run with --update-baseline to create one")
        return 0

    previous = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    regressions = []
    for name, metrics in results.items():
        old = previous.get("fixtures", {}).get(name, {}).get("onset_pitch_f1")
        if old is not None and metrics["onset_pitch_f1"] < old - REGRESSION_TOLERANCE:
            regressions.append(
                f"{name}: {metrics['onset_pitch_f1']:.3f} < {old:.3f} - {REGRESSION_TOLERANCE}"
            )
    if regressions:
        print("\nREGRESSION detected:", file=sys.stderr)
        for line in regressions:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("no regressions against baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
