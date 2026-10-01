from __future__ import annotations

import numpy as np

from music_teacher.transcription.notes import (
    NoteEvent,
    constrain_frequency,
    filter_harmonic_duplicates,
    model_frames_to_time,
    model_output_to_notes,
)


def _activation(pitch: int, start: int = 5, length: int = 15, n_frames: int = 60) -> dict:
    frames = np.zeros((n_frames, 88))
    onsets = np.zeros((n_frames, 88))
    contours = np.zeros((n_frames, 264))
    band = pitch - 21
    frames[start : start + length, band] = 0.9
    onsets[start, band] = 0.9
    contours[start : start + length, 117] = 0.9
    return {"note": frames, "onset": onsets, "contour": contours}


def test_single_note_detected() -> None:
    events = model_output_to_notes(
        _activation(60), minimum_note_length_ms=100, melodia_trick=False
    )
    assert len(events) == 1
    note = events[0]
    assert note.pitch == 60
    assert note.start == float(model_frames_to_time(60)[5])
    assert note.velocity > 0.8


def test_short_note_rejected() -> None:
    events = model_output_to_notes(
        _activation(60, length=3), minimum_note_length_ms=200, melodia_trick=False
    )
    assert events == []


def test_constrain_frequency_zeroes_outside_range() -> None:
    frames = np.ones((4, 88))
    onsets = np.ones((4, 88))
    frames, onsets = constrain_frequency(onsets, frames, max_freq=1000.0, min_freq=200.0)
    assert frames[:, 0].sum() == 0
    assert frames[:, -1].sum() == 0
    assert frames[:, 39].sum() > 0


def test_harmonic_filter() -> None:
    fundamental = NoteEvent(start=0.0, end=1.0, pitch=60, velocity=0.8)
    quiet_octave = NoteEvent(start=0.0, end=1.0, pitch=72, velocity=0.3)
    quiet_twelfth = NoteEvent(start=0.0, end=1.0, pitch=79, velocity=0.3)
    real_fifth = NoteEvent(start=0.0, end=1.0, pitch=67, velocity=0.3)
    loud_octave = NoteEvent(start=0.0, end=1.0, pitch=72, velocity=0.7)

    kept = filter_harmonic_duplicates([fundamental, quiet_octave, quiet_twelfth, real_fifth])
    assert kept == [fundamental, real_fifth]

    kept = filter_harmonic_duplicates([fundamental, loud_octave])
    assert kept == [fundamental, loud_octave]


def test_harmonic_filter_non_overlapping_kept() -> None:
    first = NoteEvent(start=0.0, end=0.5, pitch=60, velocity=0.8)
    later = NoteEvent(start=0.6, end=1.0, pitch=72, velocity=0.1)
    assert filter_harmonic_duplicates([first, later]) == [first, later]
