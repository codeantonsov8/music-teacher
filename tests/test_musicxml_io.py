from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from music_teacher.theory.musicxml_io import write_musicxml
from music_teacher.transcription.notes import NoteEvent


def test_write_musicxml_basic(tmp_path: Path) -> None:
    notes = [
        NoteEvent(start=0.0, end=0.5, pitch=60),
        NoteEvent(start=0.5, end=1.0, pitch=62),
        NoteEvent(start=1.0, end=2.0, pitch=64),
    ]
    path = write_musicxml(tmp_path / "out.musicxml", notes, tempo_bpm=120)
    tree = ET.parse(path)
    root = tree.getroot()

    assert root.tag == "score-partwise"
    assert root.find("work/work-title").text == "Transcription"
    part = root.find("part")
    measures = part.findall("measure")
    assert len(measures) == 1

    pitches = []
    for note in measures[0].findall("note"):
        step = note.find("pitch/step").text
        octave = note.find("pitch/octave").text
        pitches.append(f"{step}{octave}")
    assert pitches == ["C4", "D4", "E4"]


def test_write_musicxml_chord(tmp_path: Path) -> None:
    notes = [
        NoteEvent(start=0.0, end=1.0, pitch=60),
        NoteEvent(start=0.0, end=1.0, pitch=64),
        NoteEvent(start=0.0, end=1.0, pitch=67),
    ]
    path = write_musicxml(tmp_path / "chord.musicxml", notes, tempo_bpm=120)
    root = ET.parse(path).getroot()
    note_elements = root.find("part/measure").findall("note")
    assert len(note_elements) == 3
    assert note_elements[0].find("chord") is None
    assert note_elements[1].find("chord") is not None
    assert note_elements[2].find("chord") is not None


def test_write_musicxml_rest_between_notes(tmp_path: Path) -> None:
    notes = [
        NoteEvent(start=0.0, end=0.5, pitch=60),
        NoteEvent(start=1.0, end=1.5, pitch=62),
    ]
    path = write_musicxml(tmp_path / "rest.musicxml", notes, tempo_bpm=120)
    root = ET.parse(path).getroot()
    note_elements = root.find("part/measure").findall("note")
    assert any(element.find("rest") is not None for element in note_elements)
