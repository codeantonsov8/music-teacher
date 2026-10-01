from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from music_teacher.api.app import create_app

FIXTURE = Path("tests/fixtures/audio/sine_scale.wav")


def test_health() -> None:
    client = TestClient(create_app())
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_device_endpoint() -> None:
    client = TestClient(create_app())
    data = client.get("/api/device").json()
    assert data["compute"] in {"auto", "cpu", "cuda"}
    assert data["selected"][-1] == "CPUExecutionProvider"


def test_index_page_served() -> None:
    client = TestClient(create_app())
    res = client.get("/")
    assert res.status_code == 200
    assert "Music Teacher" in res.text


@pytest.mark.slow
def test_transcribe_upload_json() -> None:
    client = TestClient(create_app())
    res = client.post(
        "/api/transcribe",
        files={"file": ("sine_scale.wav", FIXTURE.read_bytes(), "audio/wav")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 8
    assert [note["pitch"] for note in body["notes"]] == [60, 62, 64, 65, 67, 69, 71, 72]


@pytest.mark.slow
def test_transcribe_upload_musicxml() -> None:
    client = TestClient(create_app())
    res = client.post(
        "/api/transcribe?format=musicxml",
        files={"file": ("sine_scale.wav", FIXTURE.read_bytes(), "audio/wav")},
    )
    assert res.status_code == 200
    assert b"score-partwise" in res.content
    assert b"<step>C</step>" in res.content


def test_transcribe_rejects_garbage() -> None:
    client = TestClient(create_app())
    res = client.post(
        "/api/transcribe",
        files={"file": ("noise.wav", b"not audio at all", "audio/wav")},
    )
    assert res.status_code == 400
