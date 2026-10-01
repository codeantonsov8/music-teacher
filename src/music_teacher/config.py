"""Load the shared transcription configuration from ``config/transcription.toml``."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


@dataclass(frozen=True)
class ComputeConfig:
    device: str = "auto"


@dataclass(frozen=True)
class BasicPitchConfig:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length_ms: float = 127.7
    minimum_frequency: float = 32.70
    maximum_frequency: float = 1975.53
    melodia_trick: bool = True
    harmonic_filter: bool = True


@dataclass(frozen=True)
class MonophonicConfig:
    min_confidence: float = 0.5
    frame_length: int = 2048
    hop_length: int = 256


@dataclass(frozen=True)
class TranscriptionConfig:
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    basic_pitch: BasicPitchConfig = field(default_factory=BasicPitchConfig)
    monophonic: MonophonicConfig = field(default_factory=MonophonicConfig)


def find_config() -> Path | None:
    """Locate the config file: env override, then CWD, then project root."""
    env = os.environ.get("MUSIC_TEACHER_CONFIG")
    if env:
        return Path(env)
    for candidate in (
        Path.cwd() / "config" / "transcription.toml",
        PROJECT_ROOT / "config" / "transcription.toml",
    ):
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path | None = None) -> TranscriptionConfig:
    """Load config from ``path`` (or the discovered location), falling back to defaults."""
    path = path if path is not None else find_config()
    if path is None or not path.exists():
        return TranscriptionConfig()
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return TranscriptionConfig(
        compute=ComputeConfig(**data.get("compute", {})),
        basic_pitch=BasicPitchConfig(**data.get("basic_pitch", {})),
        monophonic=MonophonicConfig(**data.get("monophonic", {})),
    )
