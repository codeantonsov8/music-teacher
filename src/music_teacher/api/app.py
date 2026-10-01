"""FastAPI application factory for the local web app."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from music_teacher import __version__
from music_teacher.api.routes import router
from music_teacher.transcription.service import warmup_async

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_async()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Music Teacher", version=__version__, lifespan=lifespan)
    app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
