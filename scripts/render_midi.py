"""Render transcription results to WAV (and optionally MIDI) for auditioning.

Input may be an audio file (transcribed on the fly) or a ``.notes.json`` file
(either ground truth or engine output).

Usage:
    uv run python scripts/render_midi.py tests/fixtures/audio/flute_melody.wav
    uv run python scripts/render_midi.py notes.json --timbre piano --midi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from music_teacher.evaluation.fixtures import render_notes
from music_teacher.theory.midi_io import write_midi
from music_teacher.transcription.notes import NoteEvent
from music_teacher.transcription.onnx_engine import BasicPitchEngine


def load_events(path: Path, device: str) -> list[NoteEvent]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            NoteEvent(
                start=note["start"],
                end=note["end"],
                pitch=note["pitch"],
                velocity=note.get("velocity", note.get("amplitude", 0.8)),
            )
            for note in data["notes"]
        ]
    engine = BasicPitchEngine(device=device)
    engine.warmup()
    return engine.transcribe(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="output .wav path")
    parser.add_argument("--midi", action="store_true", help="also export a .mid file")
    parser.add_argument("--timbre", default="piano", choices=["sine", "flute", "piano"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    events = load_events(args.input, args.device)
    out_path = args.out or args.input.with_suffix(".rendered.wav")
    audio = render_notes(events, timbre=args.timbre)
    sf.write(str(out_path), audio, 22_050, subtype="PCM_16")
    print(f"rendered {len(events)} notes -> {out_path}")

    if args.midi:
        midi_path = out_path.with_suffix(".mid")
        write_midi(midi_path, events)
        print(f"wrote {midi_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
