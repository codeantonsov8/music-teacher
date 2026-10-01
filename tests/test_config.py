from pathlib import Path

from music_teacher.config import (
    BasicPitchConfig,
    TranscriptionConfig,
    load_config,
)


def test_defaults_when_missing() -> None:
    config = load_config(Path("/nonexistent/transcription.toml"))
    assert config == TranscriptionConfig()
    assert config.basic_pitch.onset_threshold == 0.5
    assert config.compute.device == "auto"


def test_load_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "transcription.toml"
    path.write_text(
        """
[compute]
device = "cuda"

[basic_pitch]
onset_threshold = 0.4
frame_threshold = 0.2
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.compute.device == "cuda"
    assert config.basic_pitch == BasicPitchConfig(onset_threshold=0.4, frame_threshold=0.2)
    assert config.monophonic.frame_length == 2048


def test_project_config_parses() -> None:
    config = load_config()
    assert config.compute.device in {"auto", "cpu", "cuda"}
    assert 0.0 < config.basic_pitch.onset_threshold < 1.0
