# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "basic-pitch==0.4.0",
#   "onnxruntime>=1.18",
#   "numpy",
#   "soundfile",
#   "librosa",
#   "setuptools<81",
# ]
# ///
"""Generate golden reference outputs with the official basic-pitch package.

The app's lean ONNX engine is validated against these files (see
``tests/test_parity.py``). Runs in an isolated Python 3.11 environment because
basic-pitch on newer Pythons pulls in TensorFlow.

Usage: uv run --no-project scripts/gen_golden.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from music_teacher.evaluation.fixtures import default_fixtures, write_fixture  # noqa: E402

AUDIO_DIR = ROOT / "tests" / "fixtures" / "audio"
GOLDEN_DIR = ROOT / "tests" / "fixtures" / "golden"


def main() -> None:
    import basic_pitch
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model, run_inference
    from basic_pitch.note_creation import model_output_to_notes

    # Force the ONNX reference model so parity is against the exact artifact we vendor.
    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    model = Model(model_path)
    assert model.model_type.name == "ONNX", f"reference is not ONNX: {model.model_type}"
    print(f"reference model: {model_path}")

    for fixture in default_fixtures():
        wav_path, _ = write_fixture(fixture, AUDIO_DIR)
        output = run_inference(str(wav_path), model)
        np.savez_compressed(
            GOLDEN_DIR / f"{fixture.name}.npz",
            note=output["note"],
            onset=output["onset"],
            contour=output["contour"],
        )

        _, note_events = model_output_to_notes(
            output,
            onset_thresh=0.5,
            frame_thresh=0.3,
            min_note_len=11,
            min_freq=32.70,
            max_freq=1975.53,
            melodia_trick=True,
            include_pitch_bends=True,
        )
        (GOLDEN_DIR / f"{fixture.name}.notes.json").write_text(
            json.dumps(
                [
                    {
                        "start": float(start),
                        "end": float(end),
                        "pitch": int(pitch),
                        "amplitude": float(amplitude),
                        "pitch_bends": [int(b) for b in bends] if bends else None,
                    }
                    for start, end, pitch, amplitude, bends in note_events
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"golden: {fixture.name} ({len(note_events)} notes)")

    import hashlib

    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (GOLDEN_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "basic_pitch": getattr(basic_pitch, "__version__", "0.4.0"),
                "model": model_path.name,
                "model_sha256": model_sha,
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {GOLDEN_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
