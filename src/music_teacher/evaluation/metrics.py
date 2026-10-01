"""Transcription metrics (dev-only; requires mir_eval)."""

from __future__ import annotations

import mir_eval
import numpy as np


def score(
    ref_intervals: np.ndarray,
    ref_pitches: np.ndarray,
    est_intervals: np.ndarray,
    est_pitches: np.ndarray,
    onset_tolerance: float = 0.05,
    pitch_tolerance: float = 50.0,
) -> dict[str, float]:
    """Onset+pitch and onset+offset+pitch precision/recall/F1."""
    onset_pitch = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=None,
        pitch_tolerance=pitch_tolerance,
    )
    with_offset = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=0.2,
        pitch_tolerance=pitch_tolerance,
    )
    return {
        "onset_pitch_precision": float(onset_pitch[0]),
        "onset_pitch_recall": float(onset_pitch[1]),
        "onset_pitch_f1": float(onset_pitch[2]),
        "offset_pitch_precision": float(with_offset[0]),
        "offset_pitch_recall": float(with_offset[1]),
        "offset_pitch_f1": float(with_offset[2]),
    }
