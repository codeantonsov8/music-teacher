"""Minimal Standard MIDI File (format 0) writer.

Small by design: enough to export transcribed note events for playback in any
DAW or notation app, without pulling in pretty_midi.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from music_teacher.transcription.notes import NoteEvent

DIVISION = 480  # ticks per quarter note
DEFAULT_PROGRAM = 4  # Electric Piano 1


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("delta time must be non-negative")
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def midi_bytes(
    notes: Sequence[NoteEvent],
    tempo_bpm: float = 120.0,
    program: int = DEFAULT_PROGRAM,
) -> bytes:
    """Serialize note events to a format-0 MIDI file."""
    ticks_per_second = tempo_bpm / 60.0 * DIVISION

    def to_ticks(seconds: float) -> int:
        return int(round(seconds * ticks_per_second))

    raw: list[tuple[int, int, bytes]] = []
    tempo_us = int(round(60_000_000 / tempo_bpm))
    raw.append((0, 0, b"\xff\x51\x03" + tempo_us.to_bytes(3, "big")))
    raw.append((0, 1, bytes([0xC0, program & 0x7F])))
    for note in notes:
        velocity = max(1, min(127, int(round(note.velocity * 127))))
        start = to_ticks(note.start)
        end = max(to_ticks(note.end), start + 1)
        raw.append((start, 2, bytes([0x90, note.pitch & 0x7F, velocity])))
        raw.append((end, 0, bytes([0x80, note.pitch & 0x7F, 0])))

    # Note-offs before note-ons at the same tick; stable within each kind.
    raw.sort(key=lambda item: (item[0], item[1]))

    track = bytearray()
    previous_tick = 0
    for tick, _, payload in raw:
        track += _varint(tick - previous_tick)
        track += payload
        previous_tick = tick
    track += _varint(0) + b"\xff\x2f\x00"  # end of track

    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
    header += (1).to_bytes(2, "big") + DIVISION.to_bytes(2, "big")
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)


def write_midi(
    path: str | Path,
    notes: Sequence[NoteEvent],
    tempo_bpm: float = 120.0,
    program: int = DEFAULT_PROGRAM,
) -> Path:
    """Write note events to a format-0 MIDI file and return the path."""
    path = Path(path)
    path.write_bytes(midi_bytes(notes, tempo_bpm, program))
    return path
