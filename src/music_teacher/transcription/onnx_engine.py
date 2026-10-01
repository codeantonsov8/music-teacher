"""Lean Basic Pitch inference on ONNX Runtime.

Runs the vendored model directly, without TensorFlow/librosa/pretty_midi.
Pre/post-processing mirrors ``basic_pitch.inference`` for parity with the
reference oracle (see ``scripts/gen_golden.py``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

from music_teacher.config import PROJECT_ROOT, BasicPitchConfig, load_config
from music_teacher.transcription.audio import SAMPLE_RATE, load_audio
from music_teacher.transcription.device import select_providers
from music_teacher.transcription.notes import (
    ANNOTATIONS_FPS,
    NoteEvent,
    filter_harmonic_duplicates,
    model_output_to_notes,
)

log = logging.getLogger(__name__)

MODEL_FILENAME = "basic_pitch_nmp.onnx"
MODEL_SHA256 = "2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec"

AUDIO_N_SAMPLES = SAMPLE_RATE * 2 - 256  # 43_844
N_OVERLAPPING_FRAMES = 30
OVERLAP_LEN = N_OVERLAPPING_FRAMES * 256
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN
N_OLAP = N_OVERLAPPING_FRAMES // 2

_INPUT_NAME = "serving_default_input_2:0"
_OUTPUT_NAMES = {
    "note": "StatefulPartitionedCall:1",
    "onset": "StatefulPartitionedCall:2",
    "contour": "StatefulPartitionedCall:0",
}


class ModelNotFoundError(FileNotFoundError):
    pass


class ModelIntegrityError(ValueError):
    pass


def find_model_path() -> Path:
    """Locate the vendored ONNX model: env override, project, then CWD."""
    env = os.environ.get("MUSIC_TEACHER_MODEL")
    candidates = [
        Path(env) if env else None,
        PROJECT_ROOT / "models" / MODEL_FILENAME,
        Path.cwd() / "models" / MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise ModelNotFoundError(
        f"{MODEL_FILENAME} not found; expected in {PROJECT_ROOT / 'models'} "
        "or set MUSIC_TEACHER_MODEL"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows(audio: np.ndarray):
    """Yield fixed-size windows, matching basic-pitch's padding and hop exactly."""
    padded = np.concatenate([np.zeros(OVERLAP_LEN // 2, dtype=np.float32), audio])
    for start in range(0, padded.shape[0], HOP_SIZE):
        window = padded[start : start + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, (0, AUDIO_N_SAMPLES - len(window)))
        yield window


def _unwrap(outputs: list[np.ndarray], original_length: int) -> np.ndarray:
    """Merge per-window outputs into a single time-major matrix."""
    stacked = np.concatenate(outputs, axis=0)  # (n_windows, n_frames, n_freqs)
    if N_OLAP > 0:
        stacked = stacked[:, N_OLAP:-N_OLAP, :]
    shape = stacked.shape
    flat = stacked.reshape(shape[0] * shape[1], shape[2])
    n_frames = int(np.floor(original_length * (ANNOTATIONS_FPS / SAMPLE_RATE)))
    return flat[:n_frames, :]


class BasicPitchEngine:
    """Audio -> note events using the vendored Basic Pitch ONNX model."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        config: BasicPitchConfig | None = None,
        device: str | None = None,
        *,
        verify: bool = True,
    ) -> None:
        self.config = config or load_config().basic_pitch
        device = device or load_config().compute.device
        self.model_path = Path(model_path) if model_path else find_model_path()
        if verify and os.environ.get("MUSIC_TEACHER_SKIP_MODEL_CHECK") != "1":
            actual = file_sha256(self.model_path)
            if actual != MODEL_SHA256:
                raise ModelIntegrityError(
                    f"model hash mismatch: {actual} != {MODEL_SHA256} ({self.model_path})"
                )
        self.providers = select_providers(device)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path), providers=self.providers, sess_options=options
        )
        self.device = device
        self._warm = False
        self._warm_lock = threading.Lock()

    @property
    def active_providers(self) -> list[str]:
        return list(self.session.get_providers())

    def warmup(self) -> None:
        """Run one silent window so the first real request isn't a cold start."""
        with self._warm_lock:
            if self._warm:
                return
            dummy = np.zeros((1, AUDIO_N_SAMPLES, 1), dtype=np.float32)
            self.session.run(list(_OUTPUT_NAMES.values()), {_INPUT_NAME: dummy})
            self._warm = True

    def predict(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """Run the model on mono float32 audio at 22 050 Hz.

        Returns note/onset/contour matrices shaped ``(n_frames, n_freqs)``.
        """
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"expected mono audio, got shape {audio.shape}")
        collected: dict[str, list[np.ndarray]] = {key: [] for key in _OUTPUT_NAMES}
        for window in _windows(audio):
            batch = window[np.newaxis, :, np.newaxis]
            values = self.session.run(list(_OUTPUT_NAMES.values()), {_INPUT_NAME: batch})
            for key, value in zip(_OUTPUT_NAMES, values, strict=True):
                collected[key].append(value)
        return {
            key: _unwrap(values, audio.shape[0]) for key, values in collected.items()
        }

    def transcribe(self, path: str | Path) -> list[NoteEvent]:
        """Transcribe an audio file into note events."""
        return self.notes_from_audio(load_audio(path))

    def notes_from_audio(self, audio: np.ndarray) -> list[NoteEvent]:
        output = self.predict(audio)
        events = model_output_to_notes(
            output,
            onset_threshold=self.config.onset_threshold,
            frame_threshold=self.config.frame_threshold,
            minimum_note_length_ms=self.config.minimum_note_length_ms,
            minimum_frequency=self.config.minimum_frequency,
            maximum_frequency=self.config.maximum_frequency,
            melodia_trick=self.config.melodia_trick,
        )
        if self.config.harmonic_filter:
            events = filter_harmonic_duplicates(events)
        return events
