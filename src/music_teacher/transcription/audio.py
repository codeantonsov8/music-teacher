"""Audio loading and resampling for the transcription pipeline.

Keeps the runtime lean: ``soundfile`` for decoding, ``scipy`` for resampling.
"""

from __future__ import annotations

from io import BytesIO
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 22_050


class AudioLoadError(ValueError):
    """Raised when an audio file cannot be decoded."""


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono float32 signal with a polyphase filter."""
    if orig_sr == target_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def _decode(source, sample_rate: int, label: str) -> np.ndarray:
    try:
        data, sr = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioLoadError(f"could not decode {label}: {exc}") from exc
    if data.size == 0:
        raise AudioLoadError(f"empty audio: {label}")
    mono = data.mean(axis=1, dtype=np.float32)
    return resample(mono, sr, sample_rate)


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as mono float32 at ``sample_rate``.

    Raises:
        AudioLoadError: if the file is missing, empty, or undecodable.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"not a file: {path}")
    return _decode(str(path), sample_rate, str(path))


def load_audio_bytes(data: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode in-memory audio bytes as mono float32 at ``sample_rate``."""
    if not data:
        raise AudioLoadError("empty upload")
    return _decode(BytesIO(data), sample_rate, "<upload>")


def to_float32_mono(data: np.ndarray) -> np.ndarray:
    """Coerce an in-memory buffer (samples, channels) or (samples,) to mono float32."""
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim != 1:
        raise AudioLoadError(f"unsupported audio shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)
