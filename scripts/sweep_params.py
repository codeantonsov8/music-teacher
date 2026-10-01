"""Grid-search transcription post-processing parameters against the fixtures.

Usage: uv run python scripts/sweep_params.py [--top 10]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from music_teacher.config import load_config
from music_teacher.evaluation.fixtures import default_fixtures, write_fixture
from music_teacher.evaluation.metrics import score
from music_teacher.transcription.audio import load_audio
from music_teacher.transcription.notes import (
    filter_harmonic_duplicates,
    model_output_to_notes,
)
from music_teacher.transcription.onnx_engine import BasicPitchEngine

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "tests" / "fixtures" / "audio"


def load_reference(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads((AUDIO_DIR / f"{name}.notes.json").read_text(encoding="utf-8"))
    intervals = np.array([[n["start"], n["end"]] for n in data["notes"]], dtype=float)
    pitches = np.array([n["pitch"] for n in data["notes"]], dtype=float)
    return intervals, pitches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    for fixture in default_fixtures():
        write_fixture(fixture, AUDIO_DIR)
    names = [fixture.name for fixture in default_fixtures()]
    references = {name: load_reference(name) for name in names}

    base = load_config().basic_pitch
    engine = BasicPitchEngine(config=base, device="cpu")
    engine.warmup()
    outputs = {
        name: engine.predict(load_audio(AUDIO_DIR / f"{name}.wav")) for name in names
    }

    rows = []
    for onset in (0.3, 0.4, 0.5, 0.6):
        for frame in (0.2, 0.3, 0.4):
            for harmonic in (True, False):
                config = dataclasses.replace(
                    base,
                    onset_threshold=onset,
                    frame_threshold=frame,
                    harmonic_filter=harmonic,
                )
                f1s = []
                for name in names:
                    events = model_output_to_notes(
                        outputs[name],
                        onset_threshold=config.onset_threshold,
                        frame_threshold=config.frame_threshold,
                        minimum_note_length_ms=config.minimum_note_length_ms,
                        minimum_frequency=config.minimum_frequency,
                        maximum_frequency=config.maximum_frequency,
                        melodia_trick=config.melodia_trick,
                    )
                    if harmonic:
                        events = filter_harmonic_duplicates(events)
                    intervals = np.array([[e.start, e.end] for e in events], dtype=float)
                    pitches = np.array([e.pitch for e in events], dtype=float)
                    if intervals.size == 0:
                        intervals = np.zeros((0, 2))
                        pitches = np.zeros((0,))
                    ref_intervals, ref_pitches = references[name]
                    metrics = score(ref_intervals, ref_pitches, intervals, pitches)
                    f1s.append(metrics["onset_pitch_f1"])
                rows.append(
                    {
                        "onset_threshold": onset,
                        "frame_threshold": frame,
                        "harmonic_filter": harmonic,
                        "mean_f1": float(np.mean(f1s)),
                    }
                )

    rows.sort(key=lambda row: row["mean_f1"], reverse=True)
    print(f"{'onset':>6} {'frame':>6} {'harm.':>6} {'mean F1':>8}")
    for row in rows[: args.top]:
        print(
            f"{row['onset_threshold']:>6.2f} {row['frame_threshold']:>6.2f} "
            f"{str(row['harmonic_filter']):>6} {row['mean_f1']:>8.4f}"
        )
    print(f"\nbest mean F1: {rows[0]['mean_f1']:.4f} (of {len(rows)} combinations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
