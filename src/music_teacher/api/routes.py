"""API routes."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from music_teacher import __version__
from music_teacher.config import load_config
from music_teacher.theory import midi_bytes, musicxml_bytes
from music_teacher.transcription.audio import AudioLoadError, load_audio_bytes
from music_teacher.transcription.device import describe
from music_teacher.transcription.service import get_engine

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/device")
def device() -> dict[str, object]:
    config = load_config()
    return {"compute": config.compute.device, **describe(config.compute.device)}


@router.post("/transcribe")
async def transcribe(
    file: Annotated[UploadFile, File()],
    format: Annotated[str, Query(pattern="^(json|musicxml|midi)$")] = "json",
    device: str | None = None,
) -> Response:
    """Transcribe an uploaded audio file into notes, MusicXML, or MIDI."""
    data = await file.read()
    try:
        audio = load_audio_bytes(data)
    except AudioLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    engine = get_engine(device)
    events = engine.notes_from_audio(audio)

    if format == "musicxml":
        return Response(
            content=musicxml_bytes(events),
            media_type="application/vnd.recordare.musicxml+xml",
        )
    if format == "midi":
        return Response(content=midi_bytes(events), media_type="audio/midi")

    return Response(
        content=json.dumps(
            {
                "count": len(events),
                "duration": round(audio.shape[0] / 22_050, 3),
                "providers": engine.active_providers,
                "notes": [
                    {
                        "start": round(event.start, 4),
                        "end": round(event.end, 4),
                        "pitch": event.pitch,
                        "velocity": round(event.velocity, 4),
                    }
                    for event in events
                ],
            }
        ),
        media_type="application/json",
    )
