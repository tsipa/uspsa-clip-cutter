"""Timer-beep detection via spectrogram energy analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

log = logging.getLogger(__name__)


@dataclass
class BeepCandidate:
    timestamp: float
    energy: float
    confidence: float


def detect_beeps(
    wav_path: Path,
    search_start: float,
    search_end: float,
    freq_low: float = 2500.0,
    freq_high: float = 5000.0,
    min_duration: float = 0.05,
    collapse_window: float = 0.3,
) -> list[BeepCandidate]:
    """Detect short high-frequency beeps in *wav_path* between *search_start* and *search_end*.

    The detector computes a spectrogram, isolates the *freq_low*--*freq_high* band,
    and looks for energy spikes that stand out from the local background.
    """
    sample_rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)

    start_sample = max(0, int(search_start * sample_rate))
    end_sample = min(len(data), int(search_end * sample_rate))

    if end_sample <= start_sample:
        log.warning("Beep search window is empty (%.2f–%.2f s)", search_start, search_end)
        return []

    segment = data[start_sample:end_sample]

    nperseg = min(1024, len(segment))
    if nperseg < 64:
        log.warning("Audio segment too short for beep detection")
        return []

    noverlap = nperseg // 2
    freqs, times, Sxx = spectrogram(
        segment, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
    )

    band_mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not band_mask.any():
        log.warning("No frequency bins in %.0f–%.0f Hz range", freq_low, freq_high)
        return []

    band_energy = Sxx[band_mask, :].mean(axis=0)

    if band_energy.max() == 0:
        return []

    median_energy = float(np.median(band_energy))
    std_energy = float(np.std(band_energy))
    threshold = median_energy + 3.0 * std_energy

    candidates: list[BeepCandidate] = []
    for i, e in enumerate(band_energy):
        if e >= threshold:
            abs_time = search_start + float(times[i])
            confidence = min(1.0, float(e / (median_energy + std_energy + 1e-9)))
            candidates.append(BeepCandidate(
                timestamp=abs_time,
                energy=float(e),
                confidence=confidence,
            ))

    candidates = _collapse(candidates, collapse_window)
    candidates.sort(key=lambda c: c.timestamp)

    log.info(
        "Beep detection: searched %.2f–%.2f s, threshold=%.2f (median=%.2f + 3*std=%.2f), found %d candidates",
        search_start, search_end, threshold, median_energy, std_energy, len(candidates),
    )
    for c in candidates:
        log.info(
            "  BEEP candidate: t=%.3fs energy=%.2f confidence=%.3f",
            c.timestamp, c.energy, c.confidence,
        )
    if not candidates:
        log.warning("  No beep detected in window %.2f–%.2f s", search_start, search_end)

    return candidates


def _collapse(candidates: list[BeepCandidate], window: float) -> list[BeepCandidate]:
    """Merge candidates that are within *window* seconds, keeping highest energy."""
    if not candidates:
        return candidates
    candidates.sort(key=lambda c: c.timestamp)
    merged: list[BeepCandidate] = [candidates[0]]
    for c in candidates[1:]:
        if c.timestamp - merged[-1].timestamp < window:
            if c.energy > merged[-1].energy:
                merged[-1] = c
        else:
            merged.append(c)
    return merged
