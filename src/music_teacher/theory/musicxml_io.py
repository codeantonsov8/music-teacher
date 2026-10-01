"""Minimal MusicXML (score-partwise) writer.

Quantizes transcribed note events to a fixed grid and emits a single-voice,
single-part score. This is intentionally a narrow subset: enough for the app's
score view and external notation tools, without music21 in the runtime.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from music_teacher.transcription.notes import NoteEvent

DIVISIONS_PER_QUARTER = 4  # 16th-note resolution
_DURATION_TYPES = {
    16: "whole",
    12: "half",
    8: "half",
    6: "quarter",
    4: "quarter",
    3: "eighth",
    2: "eighth",
    1: "16th",
}
_STEP_NAMES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
_ALTERS = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]


def _pitch_element(pitch: int) -> ET.Element:
    octave = pitch // 12 - 1
    pitch_class = pitch % 12
    element = ET.Element("pitch")
    ET.SubElement(element, "step").text = _STEP_NAMES[pitch_class]
    alter = _ALTERS[pitch_class]
    if alter:
        ET.SubElement(element, "alter").text = str(alter)
    ET.SubElement(element, "octave").text = str(octave)
    return element


def _note_element(pitch: int, duration: int, *, chord: bool = False) -> ET.Element:
    note = ET.Element("note")
    if chord:
        ET.SubElement(note, "chord")
    note.append(_pitch_element(pitch))
    ET.SubElement(note, "duration").text = str(duration)
    note_type = _DURATION_TYPES.get(duration)
    if note_type:
        ET.SubElement(note, "type").text = note_type
    return note


def _rest_element(duration: int) -> ET.Element:
    note = ET.Element("note")
    ET.SubElement(note, "rest")
    ET.SubElement(note, "duration").text = str(duration)
    note_type = _DURATION_TYPES.get(duration)
    if note_type:
        ET.SubElement(note, "type").text = note_type
    return note


def musicxml_bytes(
    notes: Sequence[NoteEvent],
    tempo_bpm: float = 120.0,
    title: str = "Transcription",
    beats_per_measure: int = 4,
    beat_type: int = 4,
) -> bytes:
    """Serialize note events to a quantized MusicXML score."""
    grid_seconds = 60.0 / tempo_bpm / DIVISIONS_PER_QUARTER
    measure_length = beats_per_measure * DIVISIONS_PER_QUARTER

    groups: dict[int, list[NoteEvent]] = {}
    for note in notes:
        start = max(0, int(round(note.start / grid_seconds)))
        groups.setdefault(start, []).append(note)

    end_division = 0
    for note in notes:
        end_division = max(end_division, int(round(note.end / grid_seconds)))

    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Music Teacher"
    part = ET.SubElement(root, "part", id="P1")

    measure_index = 0
    measure_start = 0
    while measure_start < max(end_division, 1):
        measure_end = measure_start + measure_length
        measure = ET.SubElement(part, "measure", number=str(measure_index + 1))

        if measure_index == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(DIVISIONS_PER_QUARTER)
            ET.SubElement(attributes, "key").append(_fifths_element())
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = str(beats_per_measure)
            ET.SubElement(time, "beat-type").text = str(beat_type)
            clef = ET.SubElement(attributes, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = str(int(round(tempo_bpm)))

        position = measure_start
        while position < measure_end and position < max(end_division, 1):
            if position in groups:
                pitches = sorted(note.pitch for note in groups[position])
                duration = max(
                    int(round(note.end / grid_seconds)) - position for note in groups[position]
                )
                duration = max(1, min(duration, measure_end - position))
                for index, pitch in enumerate(pitches):
                    measure.append(_note_element(pitch, duration, chord=index > 0))
                position += duration
            else:
                next_starts = [start for start in groups if position < start < measure_end]
                target = min(next_starts) if next_starts else measure_end
                target = min(target, measure_end, max(end_division, 1))
                if target <= position:
                    break
                measure.append(_rest_element(target - position))
                position = target

        measure_start = measure_end
        measure_index += 1

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
        + xml_bytes.split(b"\n", 1)[1]
    )


def write_musicxml(
    path: str | Path,
    notes: Sequence[NoteEvent],
    tempo_bpm: float = 120.0,
    title: str = "Transcription",
    beats_per_measure: int = 4,
    beat_type: int = 4,
) -> Path:
    """Write note events as a quantized MusicXML score and return the path."""
    path = Path(path)
    path.write_bytes(musicxml_bytes(notes, tempo_bpm, title, beats_per_measure, beat_type))
    return path


def _fifths_element() -> ET.Element:
    element = ET.Element("fifths")
    element.text = "0"
    return element
