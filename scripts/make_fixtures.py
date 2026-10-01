"""Generate the synthetic fixture set (audio + ground truth) for the lab and tests.

Usage: uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from music_teacher.evaluation.fixtures import default_fixtures, write_fixture

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    output_dir = ROOT / "tests" / "fixtures" / "audio"
    for fixture in default_fixtures():
        wav_path, notes_path = write_fixture(fixture, output_dir)
        print(f"wrote {wav_path.relative_to(ROOT)} and {notes_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
