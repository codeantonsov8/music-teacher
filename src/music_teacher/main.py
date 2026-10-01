"""CLI entry point: ``music-teacher serve`` and ``music-teacher transcribe``."""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from threading import Timer


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    if not args.no_browser:
        Timer(1.0, webbrowser.open, args=(f"http://{args.host}:{args.port}",)).start()
    uvicorn.run(
        "music_teacher.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def _transcribe(args: argparse.Namespace) -> None:
    from music_teacher.theory import write_midi, write_musicxml
    from music_teacher.transcription.service import get_engine

    source = Path(args.file)
    if not source.is_file():
        raise SystemExit(f"not a file: {source}")

    engine = get_engine(args.device)
    engine.warmup()
    events = engine.transcribe(source)

    out_dir = Path(args.out) if args.out else source.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    notes_path = out_dir / f"{stem}.notes.json"
    midi_path = out_dir / f"{stem}.mid"
    xml_path = out_dir / f"{stem}.musicxml"

    notes_path.write_text(
        json.dumps(
            {
                "source": str(source),
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_midi(midi_path, events)
    write_musicxml(xml_path, events)

    print(f"transcribed {source.name}: {len(events)} notes")
    print(f"  providers: {', '.join(engine.active_providers)}")
    for path in (notes_path, midi_path, xml_path):
        print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="music-teacher", description="Learn to play any music.")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the local web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true")
    serve.add_argument("--reload", action="store_true")

    transcribe = sub.add_parser("transcribe", help="Transcribe an audio file")
    transcribe.add_argument("file", help="path to an audio file")
    transcribe.add_argument("--out", default=None, help="output directory")
    transcribe.add_argument("--device", default=None, help="auto | cpu | cuda")

    args = parser.parse_args()
    if args.command == "serve":
        _serve(args)
    elif args.command == "transcribe":
        _transcribe(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
