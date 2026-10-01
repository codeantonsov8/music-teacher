"""Decode Basic Pitch model outputs into note events.

This is a dependency-light port of ``basic_pitch.note_creation`` (Apache-2.0,
Spotify AB) that avoids librosa/pretty_midi/mir_eval in the runtime. The
reference package is used in the dev lab to validate parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import argrelmax, windows

MIDI_OFFSET = 21
MAX_FREQ_IDX = 87
N_PITCH_BEND_TICKS = 8192
CONTOURS_BINS_PER_SEMITONE = 3

AUDIO_SAMPLE_RATE = 22_050
FFT_HOP = 256
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
ANNOT_N_FRAMES = ANNOTATIONS_FPS * 2
N_FREQ_BINS_CONTOURS = 88 * CONTOURS_BINS_PER_SEMITONE


@dataclass(frozen=True)
class NoteEvent:
    """A transcribed note. Times in seconds, ``velocity`` in 0..1."""

    start: float
    end: float
    pitch: int
    velocity: float = 1.0
    pitch_bends: tuple[int, ...] | None = field(default=None, compare=False)

    @property
    def duration(self) -> float:
        return self.end - self.start


def hz_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def model_frames_to_time(n_frames: int) -> np.ndarray:
    """Frame index -> seconds, including the basic-pitch window offset correction."""
    frames = np.arange(n_frames)
    times = frames * (FFT_HOP / AUDIO_SAMPLE_RATE)
    window_numbers = np.floor(frames / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (ANNOT_N_FRAMES - 43_844 / FFT_HOP) + 0.0018
    return times - (window_offset * window_numbers)


def constrain_frequency(
    onsets: np.ndarray,
    frames: np.ndarray,
    max_freq: float | None,
    min_freq: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    onsets = onsets.copy()
    frames = frames.copy()
    if max_freq is not None:
        max_idx = int(np.round(hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, max_idx:] = 0
        frames[:, max_idx:] = 0
    if min_freq is not None:
        min_idx = int(np.round(hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :min_idx] = 0
        frames[:, :min_idx] = 0
    return onsets, frames


def get_infered_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    """Augment predicted onsets with large frame-energy increases (reference spelling kept)."""
    diffs = []
    for n in range(1, n_diff + 1):
        appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(appended[n:, :] - appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    max_diff = np.max(frame_diff)
    if max_diff > 0:
        frame_diff = np.max(onsets) * frame_diff / max_diff
    return np.maximum(onsets, frame_diff)


def output_to_notes_polyphonic(
    frames: np.ndarray,
    onsets: np.ndarray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    max_freq: float | None,
    min_freq: float | None,
    infer_onsets: bool = True,
    melodia_trick: bool = True,
    energy_tol: int = 11,
) -> list[tuple[int, int, int, float]]:
    """Decode activation matrices to ``(start_frame, end_frame, midi_pitch, amplitude)``."""
    n_frames = frames.shape[0]
    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]
    onset_freq_idx = onset_idx[1][::-1]

    remaining_energy = frames.copy()
    note_events: list[tuple[int, int, int, float]] = []

    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx, strict=True):
        if note_start_idx >= n_frames - 1:
            continue
        i = note_start_idx + 1
        k = 0
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1
        i -= k
        if i - note_start_idx <= min_note_len:
            continue
        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0
        amplitude = float(np.mean(frames[note_start_idx:i, freq_idx]))
        note_events.append((note_start_idx, i, int(freq_idx) + MIDI_OFFSET, amplitude))

    if melodia_trick:
        energy_shape = remaining_energy.shape
        while np.max(remaining_energy) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0

            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i += 1
            i_end = i - 1 - k

            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i -= 1
            i_start = i + 1 + k

            if i_end - i_start <= min_note_len:
                continue
            amplitude = float(np.mean(frames[i_start:i_end, freq_idx]))
            note_events.append((i_start, i_end, int(freq_idx) + MIDI_OFFSET, amplitude))

    return note_events


def midi_pitch_to_contour_bin(pitch_midi: int) -> float:
    pitch_hz = 440.0 * 2.0 ** ((pitch_midi - 69) / 12.0)
    return 12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(pitch_hz / 27.5)


def get_pitch_bends(
    contours: np.ndarray,
    note_events: list[tuple[int, int, int, float]],
    n_bins_tolerance: int = 25,
) -> list[tuple[int, int, int, float, list[int]]]:
    window_length = n_bins_tolerance * 2 + 1
    freq_gaussian = windows.gaussian(window_length, std=5)
    result = []
    for start_idx, end_idx, pitch_midi, amplitude in note_events:
        freq_idx = int(np.round(midi_pitch_to_contour_bin(pitch_midi)))
        freq_start_idx = max(freq_idx - n_bins_tolerance, 0)
        freq_end_idx = min(N_FREQ_BINS_CONTOURS, freq_idx + n_bins_tolerance + 1)
        gaussian_slice = freq_gaussian[
            max(0, n_bins_tolerance - freq_idx) : window_length
            - max(0, freq_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1))
        ]
        submatrix = contours[start_idx:end_idx, freq_start_idx:freq_end_idx] * gaussian_slice
        pb_shift = n_bins_tolerance - max(0, n_bins_tolerance - freq_idx)
        bends = list(np.argmax(submatrix, axis=1) - pb_shift)
        result.append((start_idx, end_idx, pitch_midi, amplitude, bends))
    return result


def model_output_to_notes(
    output: dict[str, np.ndarray],
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 127.7,
    minimum_frequency: float | None = 32.70,
    maximum_frequency: float | None = 1975.53,
    melodia_trick: bool = True,
    include_pitch_bends: bool = True,
) -> list[NoteEvent]:
    """Convert model activation matrices to time-stamped note events."""
    frames = output["note"]
    onsets = output["onset"]
    contours = output["contour"]

    min_note_len = int(
        np.round(minimum_note_length_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))
    )
    raw = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,
        max_freq=maximum_frequency,
        min_freq=minimum_frequency,
        melodia_trick=melodia_trick,
    )
    with_bends = get_pitch_bends(contours, raw) if include_pitch_bends else None
    times = model_frames_to_time(contours.shape[0])

    events = []
    for index, (start_idx, end_idx, pitch, amplitude) in enumerate(raw):
        bends = tuple(with_bends[index][4]) if with_bends is not None else None
        events.append(
            NoteEvent(
                start=float(times[start_idx]),
                end=float(times[end_idx]),
                pitch=int(pitch),
                velocity=float(amplitude),
                pitch_bends=bends,
            )
        )
    events.sort(key=lambda note: (note.start, note.pitch))
    return events


_HARMONIC_ORDERS = (2, 3, 4, 5)


def filter_harmonic_duplicates(
    events: list[NoteEvent],
    velocity_ratio: float = 0.65,
    semitone_tolerance: float = 0.6,
) -> list[NoteEvent]:
    """Drop quiet notes that duplicate a louder note's harmonic series.

    Harmonic-rich instruments (flute, piano) make the model report partials as
    separate, much quieter notes: octave (h=2, +12), twelfth (h=3, +19), double
    octave (h=4, +24), and so on. Only near-integer harmonic ratios are matched,
    so genuine intervals like fifths are preserved. Applied iteratively so
    stacked partials collapse onto the fundamental.
    """
    ratios = [(order, 12.0 * float(np.log2(order))) for order in _HARMONIC_ORDERS]
    kept = sorted(events, key=lambda note: (note.start, note.pitch))
    while True:
        drop: set[int] = set()
        for i, fundamental in enumerate(kept):
            if i in drop:
                continue
            for j, partial in enumerate(kept):
                if i == j or j in drop:
                    continue
                if not (partial.start < fundamental.end and fundamental.start < partial.end):
                    continue
                interval = partial.pitch - fundamental.pitch
                is_harmonic = any(
                    abs(interval - semitones) <= semitone_tolerance for _, semitones in ratios
                )
                if is_harmonic and partial.velocity < velocity_ratio * fundamental.velocity:
                    drop.add(j)
        if not drop:
            return kept
        kept = [note for index, note in enumerate(kept) if index not in drop]


def drop_overlapping_pitch_bends(events: list[NoteEvent]) -> list[NoteEvent]:
    """Remove pitch bends from any notes that overlap in time with another note."""
    ordered = sorted(events, key=lambda n: (n.start, n.end))
    mutable = [n for n in ordered]
    for i in range(len(mutable) - 1):
        for j in range(i + 1, len(mutable)):
            if mutable[j].start >= mutable[i].end:
                break
            mutable[i] = NoteEvent(**{**mutable[i].__dict__, "pitch_bends": None})
            mutable[j] = NoteEvent(**{**mutable[j].__dict__, "pitch_bends": None})
    return mutable
