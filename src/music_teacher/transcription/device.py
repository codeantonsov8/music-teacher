"""ONNX Runtime execution-provider selection.

Order of preference for ``auto``: CUDA -> CoreML -> CPU. A requested device that
is unavailable degrades to CPU with a warning instead of failing.

CUDA wheels installed from PyPI need their bundled cuDNN/cuBLAS libraries
preloaded before a session can be created; ``prepare_runtime`` handles that.
"""

from __future__ import annotations

import logging

import onnxruntime as ort

log = logging.getLogger(__name__)

_PREFERRED = ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
_VALID_DEVICES = ("auto", "cpu", "cuda")

_preloaded = False


def prepare_runtime() -> None:
    """Preload CUDA/cuDNN libraries shipped as PyPI wheels (idempotent, best-effort)."""
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception as exc:  # pragma: no cover - depends on platform
        log.warning("Could not preload CUDA libraries: %s", exc)


def available_providers() -> list[str]:
    return list(ort.get_available_providers())


def select_providers(device: str = "auto") -> list[str]:
    """Return the provider list for an ``InferenceSession``."""
    if device not in _VALID_DEVICES:
        log.warning("Unknown device %r, using 'auto'", device)
        device = "auto"
    available = set(available_providers())
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" in available:
            prepare_runtime()
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        log.warning("CUDA requested but unavailable; falling back to CPU")
        return ["CPUExecutionProvider"]
    selected = [p for p in _PREFERRED if p in available]
    if selected and selected[0] != "CPUExecutionProvider":
        prepare_runtime()
    return selected or ["CPUExecutionProvider"]


def describe(device: str = "auto") -> dict[str, object]:
    """Human-readable provider status for diagnostics and the API."""
    return {
        "requested": device,
        "available": available_providers(),
        "selected": select_providers(device),
    }
