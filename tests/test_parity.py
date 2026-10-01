"""Parity between the lean ONNX engine and the basic-pitch reference oracle.

Golden files are produced by ``uv run --no-project scripts/gen_golden.py``.
Tests are skipped when the golden fixtures are absent so the suite still runs
on a fresh checkout before the lab has been bootstrapped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from music_teacher.transcription.audio import load_audio
from music_teacher.transcription.device import available_providers
from music_teacher.transcription.notes import model_output_to_notes
from music_teacher.transcription.onnx_engine import BasicPitchEngine

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "tests" / "fixtures" / "audio"
GOLDEN_DIR = ROOT / "tests" / "fixtures" / "golden"

NAMES = ["sine_a4", "sine_scale", "flute_melody", "flute_noisy", "piano_chords"]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not GOLDEN_DIR.is_dir() or not (GOLDEN_DIR / "manifest.json").is_file(),
        reason="golden fixtures missing; run scripts/gen_golden.py",
    ),
]


@pytest.fixture(scope="module")
def engine() -> BasicPitchEngine:
    instance = BasicPitchEngine(device="cpu")
    return instance


def _reference_notes(name: str) -> list[tuple]:
    data = json.loads((GOLDEN_DIR / f"{name}.notes.json").read_text(encoding="utf-8"))
    return sorted(
        (round(item["start"], 9), round(item["end"], 9), item["pitch"], round(item["amplitude"], 6))
        for item in data
    )


@pytest.mark.parametrize("name", NAMES)
def test_posteriorgram_parity(engine: BasicPitchEngine, name: str) -> None:
    golden = np.load(GOLDEN_DIR / f"{name}.npz")
    output = engine.predict(load_audio(AUDIO_DIR / f"{name}.wav"))
    for key in ("note", "onset", "contour"):
        assert output[key].shape == golden[key].shape, key
        np.testing.assert_allclose(output[key], golden[key], atol=1e-5, err_msg=key)


@pytest.mark.parametrize("name", NAMES)
def test_note_event_parity(engine: BasicPitchEngine, name: str) -> None:
    output = engine.predict(load_audio(AUDIO_DIR / f"{name}.wav"))
    events = model_output_to_notes(output)
    ours = sorted(
        (round(event.start, 9), round(event.end, 9), event.pitch, round(event.velocity, 6))
        for event in events
    )
    assert ours == _reference_notes(name)


@pytest.mark.gpu
@pytest.mark.skipif(
    "CUDAExecutionProvider" not in available_providers(), reason="CUDA unavailable"
)
def test_gpu_matches_cpu_within_tolerance() -> None:
    cpu = BasicPitchEngine(device="cpu")
    gpu = BasicPitchEngine(device="cuda")
    assert "CUDAExecutionProvider" in gpu.active_providers
    audio = load_audio(AUDIO_DIR / "flute_melody.wav")
    expected = cpu.predict(audio)
    actual = gpu.predict(audio)
    for key in ("note", "onset", "contour"):
        assert actual[key].shape == expected[key].shape
        np.testing.assert_allclose(actual[key], expected[key], atol=5e-3, err_msg=key)
