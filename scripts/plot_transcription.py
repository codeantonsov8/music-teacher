"""Plot waveform, spectrogram, and detected notes for an audio file.

Usage: uv run python scripts/plot_transcription.py AUDIO [--out PATH] [--device cpu]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import librosa  # noqa: E402
import librosa.display  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from music_teacher.transcription.audio import load_audio  # noqa: E402
from music_teacher.transcription.onnx_engine import BasicPitchEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-transcribe", action="store_true", help="plot audio only")
    args = parser.parse_args()

    audio = load_audio(args.audio)
    sr = 22_050
    out_path = args.out or (ROOT / "tests" / "fixtures" / "plots" / f"{args.audio.stem}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    events = []
    if not args.no_transcribe:
        engine = BasicPitchEngine(device=args.device)
        engine.warmup()
        events = engine.transcribe(args.audio)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    times = [i / sr for i in range(audio.shape[0])]
    top.plot(times, audio, linewidth=0.5, color="#4f7cff")
    top.set_ylabel("amplitude")
    top.set_title(f"{args.audio.name} — {len(events)} detected notes")
    top.grid(alpha=0.2)

    spectrum = librosa.amplitude_to_db(
        abs(librosa.stft(audio.astype(float), n_fft=2048, hop_length=512)), ref=1.0
    )
    image = librosa.display.specshow(
        spectrum,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="log",
        ax=bottom,
        cmap="magma",
    )
    for event in events:
        frequency = librosa.midi_to_hz(event.pitch)
        bottom.plot(
            [event.start, event.end],
            [frequency, frequency],
            color="#6ee7a0",
            linewidth=2,
            alpha=0.4 + 0.6 * min(event.velocity, 1.0),
        )
    figure.colorbar(image, ax=bottom, format="%+2.0f dB", pad=0.01)
    bottom.set_title("spectrogram with detected notes")
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")
    print(f"detected {len(events)} notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
