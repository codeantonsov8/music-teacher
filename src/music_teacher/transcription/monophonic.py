"""Monophonic pitch tracking (YIN) for single-note instruments and the live path.

This is the server-side reference implementation used for benchmarks and
post-session monophonic transcription. The browser ships the same algorithm in
an AudioWorklet; keeping this here lets the lab measure its accuracy and latency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from music_teacher.transcription.audio import SAMPLE_RATE


@dataclass(frozen=True)
class PitchFrame:
    time: float
    frequency: float
    confidence: float


def _difference_function(frame: np.ndarray) -> np.ndarray:
    n = frame.shape[0]
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(frame, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]

    energy = np.concatenate([[0.0], np.cumsum(frame.astype(np.float64) ** 2)])
    taus = np.arange(n)
    diff = (energy[n - taus] - energy[0]) + (energy[n] - energy[taus]) - 2.0 * acf
    diff[0] = 0.0
    return diff


def yin_pitch(
    frame: np.ndarray,
    sr: int = SAMPLE_RATE,
    fmin: float = 65.0,
    fmax: float = 2000.0,
    threshold: float = 0.15,
) -> tuple[float, float]:
    """Estimate (frequency_hz, confidence) for one frame; frequency 0 if unvoiced."""
    frame = np.asarray(frame, dtype=np.float64)
    frame = frame - frame.mean()
    if np.max(np.abs(frame)) < 1e-6:
        return 0.0, 0.0

    tau_min = max(2, int(np.floor(sr / fmax)))
    tau_max = min(frame.shape[0] - 2, int(np.ceil(sr / fmin)))
    diff = _difference_function(frame)

    cumulative = np.cumsum(diff)
    cumulative[cumulative == 0] = 1e-12
    taus = np.arange(diff.shape[0])
    cmnd = diff * taus / cumulative
    cmnd[:tau_min] = 1.0

    search = cmnd[tau_min : tau_max + 1]
    below = np.where(search < threshold)[0]
    if below.size:
        # first local minimum under the threshold
        tau = int(below[0])
        while tau + 1 < search.shape[0] and search[tau + 1] < search[tau]:
            tau += 1
    else:
        tau = int(np.argmin(search))
    tau += tau_min

    if cmnd[tau] >= threshold and not below.size:
        # no confident candidate; still return the best if reasonably periodic
        if cmnd[tau] > 0.6:
            return 0.0, 0.0

    # parabolic interpolation around the minimum
    if 0 < tau < cmnd.shape[0] - 1:
        left, center, right = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denominator = 2 * (2 * center - left - right)
        if denominator != 0:
            tau = tau + (right - left) / denominator

    frequency = sr / tau
    confidence = float(np.clip(1.0 - cmnd[int(round(tau))], 0.0, 1.0))
    if frequency < fmin or frequency > fmax:
        return 0.0, 0.0
    return float(frequency), confidence


def track_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_length: int = 2048,
    hop_length: int = 256,
    fmin: float = 65.0,
    fmax: float = 2000.0,
    threshold: float = 0.15,
) -> list[PitchFrame]:
    """Frame-wise YIN over a signal."""
    frames = []
    for start in range(0, max(audio.shape[0] - frame_length, 0) + 1, hop_length):
        frame = audio[start : start + frame_length]
        frequency, confidence = yin_pitch(frame, sr, fmin, fmax, threshold)
        frames.append(
            PitchFrame(time=start / sr, frequency=frequency, confidence=confidence)
        )
    return frames


def frequency_to_midi(frequency: float) -> float:
    if frequency <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(frequency / 440.0)
