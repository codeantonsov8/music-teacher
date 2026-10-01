from __future__ import annotations

import numpy as np

from music_teacher.transcription.audio import SAMPLE_RATE, load_audio, resample
from music_teacher.transcription.monophonic import frequency_to_midi, track_pitch, yin_pitch


def test_resample_changes_length() -> None:
    audio = np.sin(2 * np.pi * 440 * np.arange(44_100) / 44_100).astype(np.float32)
    downsampled = resample(audio, 44_100, SAMPLE_RATE)
    assert abs(downsampled.shape[0] - 22_050) <= 1


def test_load_fixture() -> None:
    audio = load_audio("tests/fixtures/audio/sine_a4.wav")
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(audio.shape[0] / SAMPLE_RATE - 1.95) < 0.05


def test_yin_on_sine() -> None:
    t = np.arange(int(0.5 * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    frequency, confidence = yin_pitch(audio[:2048])
    assert abs(frequency - 440.0) < 2.0
    assert confidence > 0.9


def test_yin_on_silence() -> None:
    frequency, confidence = yin_pitch(np.zeros(2048, dtype=np.float32))
    assert frequency == 0.0
    assert confidence == 0.0


def test_track_pitch_finds_note() -> None:
    t = np.arange(int(0.5 * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.sin(2 * np.pi * 523.25 * t).astype(np.float32)
    frames = track_pitch(audio, frame_length=2048, hop_length=512)
    voiced = [f for f in frames if f.frequency > 0]
    assert voiced
    median = np.median([f.frequency for f in voiced])
    assert abs(median - 523.25) < 3.0
    assert abs(frequency_to_midi(median) - 72) < 0.2
