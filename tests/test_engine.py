from __future__ import annotations

from pathlib import Path

import pytest

from music_teacher.transcription.onnx_engine import BasicPitchEngine, find_model_path

pytestmark = pytest.mark.slow


def test_model_is_vendored() -> None:
    path = find_model_path()
    assert path.is_file()
    assert path.name == "basic_pitch_nmp.onnx"


def test_transcribe_scale(tmp_path: Path) -> None:
    engine = BasicPitchEngine(device="cpu")
    events = engine.transcribe("tests/fixtures/audio/sine_scale.wav")
    pitches = [event.pitch for event in events]
    assert pitches == [60, 62, 64, 65, 67, 69, 71, 72]


def test_transcribe_polyphonic_chord() -> None:
    engine = BasicPitchEngine(device="cpu")
    events = engine.transcribe("tests/fixtures/audio/piano_chords.wav")
    first_chord = sorted(event.pitch for event in events if event.start < 0.5)
    assert first_chord == [60, 64, 67]
