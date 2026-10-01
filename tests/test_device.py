from music_teacher.transcription.device import available_providers, describe, select_providers


def test_cpu_is_always_available() -> None:
    assert "CPUExecutionProvider" in available_providers()
    assert select_providers("cpu") == ["CPUExecutionProvider"]


def test_auto_ends_with_cpu_fallback() -> None:
    providers = select_providers("auto")
    assert providers[-1] == "CPUExecutionProvider"


def test_unknown_device_falls_back_to_auto() -> None:
    assert select_providers("tpu") == select_providers("auto")


def test_describe_reports_selection() -> None:
    info = describe("cpu")
    assert info["requested"] == "cpu"
    assert info["selected"] == ["CPUExecutionProvider"]
