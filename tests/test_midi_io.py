from __future__ import annotations

from pathlib import Path

from music_teacher.theory.midi_io import write_midi
from music_teacher.transcription.notes import NoteEvent


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset


def test_write_midi_structure(tmp_path: Path) -> None:
    path = write_midi(
        tmp_path / "out.mid",
        [NoteEvent(start=0.0, end=0.5, pitch=60, velocity=0.8)],
        tempo_bpm=120,
    )
    data = path.read_bytes()

    assert data[:4] == b"MThd"
    assert int.from_bytes(data[4:8], "big") == 6
    assert int.from_bytes(data[8:10], "big") == 0  # format 0
    assert int.from_bytes(data[10:12], "big") == 1  # one track
    assert int.from_bytes(data[12:14], "big") == 480  # division

    assert data[14:18] == b"MTrk"
    track_length = int.from_bytes(data[18:22], "big")
    track = data[22 : 22 + track_length]
    assert len(track) == track_length
    assert track[-3:] == b"\xff\x2f\x00"

    assert b"\xff\x51\x03\x07\xa1\x20" in track  # 120 bpm tempo
    assert bytes([0x90, 60]) in track  # note on
    assert bytes([0x80, 60]) in track  # note off

    # first event delta is 0 (tempo)
    delta, offset = _read_varint(track, 0)
    assert delta == 0
    assert track[offset] == 0xFF
