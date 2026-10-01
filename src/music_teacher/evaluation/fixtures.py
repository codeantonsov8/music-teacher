"""Synthetic audio fixtures with exact ground-truth notes.

Used by the lab, the golden generator, and the regression tests. Only numpy and
soundfile are required, so this stays importable from the lean runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 22_050


@dataclass(frozen=True)
class FixtureNote:
    start: float
    end: float
    pitch: int
    velocity: float = 0.8


@dataclass(frozen=True)
class Fixture:
    name: str
    notes: list[FixtureNote]
    timbre: str = "flute"
    noise_db: float | None = None
    trailing_silence: float = 0.25
    description: str = ""

    @property
    def duration(self) -> float:
        last = max(note.end for note in self.notes) if self.notes else 0.0
        return last + self.trailing_silence


def midi_to_hz(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def _timbre_harmonics(timbre: str) -> tuple[list[float], float, float]:
    """Return (harmonic amplitudes, attack seconds, decay tau seconds)."""
    if timbre == "sine":
        return [1.0], 0.01, 10.0
    if timbre == "flute":
        return [1.0, 0.25, 0.08, 0.03], 0.04, 10.0
    if timbre == "piano":
        return [1.0, 0.5, 0.33, 0.25, 0.2, 0.15], 0.004, 0.9
    raise ValueError(f"unknown timbre {timbre!r}")


def _render_note(pitch: int, duration: float, sr: int, timbre: str) -> np.ndarray:
    harmonics, attack, tau = _timbre_harmonics(timbre)
    freq = midi_to_hz(pitch)
    n = max(int(round(duration * sr)), 1)
    t = np.arange(n) / sr
    wave = np.zeros(n, dtype=np.float64)
    for index, amplitude in enumerate(harmonics, start=1):
        wave += amplitude * np.sin(2 * np.pi * freq * index * t)
    envelope = np.exp(-t / tau)
    attack_samples = max(int(attack * sr), 1)
    envelope[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)
    release_samples = max(int(0.02 * sr), 1)
    envelope[-release_samples:] *= np.linspace(1.0, 0.0, release_samples)
    return wave * envelope


def synthesize(fixture: Fixture, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render a fixture to mono float32 audio."""
    total = int(round(fixture.duration * sr))
    buffer = np.zeros(total, dtype=np.float64)
    for note in fixture.notes:
        tone = _render_note(note.pitch, note.end - note.start, sr, fixture.timbre)
        start = int(round(note.start * sr))
        end = min(start + tone.shape[0], total)
        buffer[start:end] += tone[: end - start] * note.velocity
    peak = np.max(np.abs(buffer))
    if peak > 0:
        buffer = buffer / peak * 0.9
    if fixture.noise_db is not None:
        rng = np.random.default_rng(0)
        signal_power = float(np.mean(buffer**2))
        noise_power = signal_power / (10 ** (fixture.noise_db / 10))
        buffer = buffer + rng.normal(0.0, np.sqrt(noise_power), buffer.shape)
    return buffer.astype(np.float32)


def render_notes(
    notes: list,
    sr: int = SAMPLE_RATE,
    timbre: str = "piano",
    trailing_silence: float = 0.25,
) -> np.ndarray:
    """Render any objects with ``start/end/pitch/velocity`` (FixtureNote or NoteEvent)."""
    duration = max((note.end for note in notes), default=0.0) + trailing_silence
    total = max(int(round(duration * sr)), 1)
    buffer = np.zeros(total, dtype=np.float64)
    for note in notes:
        tone = _render_note(note.pitch, note.end - note.start, sr, timbre)
        start = int(round(note.start * sr))
        end = min(start + tone.shape[0], total)
        if end > start:
            buffer[start:end] += tone[: end - start] * getattr(note, "velocity", 0.8)
    peak = np.max(np.abs(buffer))
    if peak > 0:
        buffer = buffer / peak * 0.9
    return buffer.astype(np.float32)


def write_fixture(fixture: Fixture, directory: Path, sr: int = SAMPLE_RATE) -> tuple[Path, Path]:
    """Write ``<name>.wav`` and ``<name>.notes.json``; return both paths."""
    directory.mkdir(parents=True, exist_ok=True)
    audio_path = directory / f"{fixture.name}.wav"
    notes_path = directory / f"{fixture.name}.notes.json"
    sf.write(str(audio_path), synthesize(fixture, sr), sr, subtype="PCM_16")
    notes_path.write_text(
        json.dumps(
            {
                "name": fixture.name,
                "sample_rate": sr,
                "timbre": fixture.timbre,
                "noise_db": fixture.noise_db,
                "description": fixture.description,
                "notes": [
                    {
                        "start": note.start,
                        "end": note.end,
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                    }
                    for note in fixture.notes
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return audio_path, notes_path


def _scale(
    start_pitch: int, steps: list[int], note_len: float, gap: float = 0.0
) -> list[FixtureNote]:
    notes = []
    time = 0.1
    for step in steps:
        notes.append(FixtureNote(start=time, end=time + note_len, pitch=start_pitch + step))
        time += note_len + gap
    return notes


def default_fixtures() -> list[Fixture]:
    """The committed fixture set: monophonic, polyphonic, and noisy cases."""
    return [
        Fixture(
            name="sine_a4",
            notes=[FixtureNote(start=0.2, end=1.7, pitch=69)],
            timbre="sine",
            description="single A4 sine tone",
        ),
        Fixture(
            name="sine_scale",
            notes=_scale(60, [0, 2, 4, 5, 7, 9, 11, 12], 0.35, 0.03),
            timbre="sine",
            description="C major scale, C4-C5",
        ),
        Fixture(
            name="flute_melody",
            notes=_scale(72, [0, 2, 4, 7, 9, 7, 4, 0], 0.4, 0.03),
            timbre="flute",
            description="C5 major-pentatonic melody, flute timbre",
        ),
        Fixture(
            name="flute_noisy",
            notes=_scale(72, [0, 2, 4, 7, 9, 7, 4, 0], 0.4, 0.03),
            timbre="flute",
            noise_db=15.0,
            description="flute melody with 15 dB SNR white noise",
        ),
        Fixture(
            name="piano_chords",
            notes=[
                FixtureNote(start=0.1, end=1.1, pitch=p)
                for p in (60, 64, 67)
            ]
            + [FixtureNote(start=1.2, end=2.2, pitch=p) for p in (65, 69, 72)]
            + [FixtureNote(start=2.3, end=3.3, pitch=p) for p in (67, 71, 74)],
            timbre="piano",
            description="C/F/G major triads, piano timbre",
        ),
    ]
