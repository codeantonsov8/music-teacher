"""Shared, lazily-created transcription engine for the API and CLI."""

from __future__ import annotations

import threading

from music_teacher.config import load_config
from music_teacher.transcription.onnx_engine import BasicPitchEngine

_lock = threading.Lock()
_engine: BasicPitchEngine | None = None


def get_engine(device: str | None = None) -> BasicPitchEngine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    with _lock:
        if _engine is None:
            config = load_config()
            _engine = BasicPitchEngine(
                config=config.basic_pitch,
                device=device or config.compute.device,
            )
    return _engine


def warmup_async() -> threading.Thread:
    """Warm the model in the background so the first request isn't a cold start."""
    thread = threading.Thread(target=lambda: get_engine().warmup(), daemon=True)
    thread.start()
    return thread
