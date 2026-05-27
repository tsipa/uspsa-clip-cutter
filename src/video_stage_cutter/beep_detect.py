"""Audio event detection: timer beeps and gunshots."""

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


@dataclass
class GunshotCandidate:
    timestamp: float
    peak_amplitude: float
    confidence: float


def load_wav(wav_path: Path) -> tuple[int, np.ndarray]:
    """Load WAV, return (sample_rate, mono float32 array)."""
    sample_rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    return sample_rate, data.astype(np.float32)


def detect_beeps(
    wav_path: Path,
    search_start: float,
    search_end: float,
    freq_low: float = 2500.0,
    freq_high: float = 5000.0,
    collapse_window: float = 0.3,
) -> list[BeepCandidate]:
    """Detect high-frequency beeps between *search_start* and *search_end*."""
    sample_rate, data = load_wav(wav_path)

    start_sample = max(0, int(search_start * sample_rate))
    end_sample = min(len(data), int(search_end * sample_rate))

    if end_sample <= start_sample:
        log.warning("Beep search window is empty (%.2f-%.2f s)", search_start, search_end)
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
        log.warning("No frequency bins in %.0f-%.0f Hz range", freq_low, freq_high)
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
                timestamp=abs_time, energy=float(e), confidence=confidence,
            ))

    candidates = _collapse_beeps(candidates, collapse_window)
    candidates.sort(key=lambda c: c.timestamp)

    log.info(
        "Beep detection: searched %.2f-%.2fs, threshold=%.2f (median=%.2f + 3*std=%.2f), found %d candidates",
        search_start, search_end, threshold, median_energy, std_energy, len(candidates),
    )
    for c in candidates:
        log.info("  BEEP candidate: t=%.3fs energy=%.2f confidence=%.3f", c.timestamp, c.energy, c.confidence)
    if not candidates:
        log.warning("  No beep detected in window %.2f-%.2fs", search_start, search_end)

    return candidates


def detect_gunshots(
    wav_path: Path,
    search_start: float = 0.0,
    search_end: float | None = None,
    collapse_window: float = 0.15,
) -> list[GunshotCandidate]:
    """Detect loud transient spikes (gunshots) in audio.

    Gunshots are broadband, very loud, and short (<0.1s rise time).
    We look for amplitude envelope spikes well above the RMS level.
    """
    sample_rate, data = load_wav(wav_path)

    if search_end is None:
        search_end = len(data) / sample_rate

    start_sample = max(0, int(search_start * sample_rate))
    end_sample = min(len(data), int(search_end * sample_rate))

    if end_sample <= start_sample:
        return []

    segment = np.abs(data[start_sample:end_sample])

    window_samples = int(0.01 * sample_rate)  # 10ms envelope
    if window_samples < 1 or len(segment) < window_samples:
        return []

    envelope = np.convolve(segment, np.ones(window_samples) / window_samples, mode="same")

    rms = float(np.sqrt(np.mean(segment ** 2)))
    if rms == 0:
        return []

    threshold = rms * 8.0

    candidates: list[GunshotCandidate] = []
    above = envelope > threshold
    i = 0
    while i < len(above):
        if above[i]:
            peak_idx = i
            peak_val = envelope[i]
            while i < len(above) and above[i]:
                if envelope[i] > peak_val:
                    peak_val = envelope[i]
                    peak_idx = i
                i += 1
            abs_time = search_start + peak_idx / sample_rate
            confidence = min(1.0, float(peak_val / (rms * 12.0)))
            candidates.append(GunshotCandidate(
                timestamp=abs_time, peak_amplitude=float(peak_val), confidence=confidence,
            ))
        else:
            i += 1

    candidates = _collapse_gunshots(candidates, collapse_window)
    candidates.sort(key=lambda c: c.timestamp)

    log.info(
        "Gunshot detection: searched %.2f-%.2fs, rms=%.1f, threshold=%.1f, found %d candidates",
        search_start, search_end, rms, threshold, len(candidates),
    )

    return candidates


def _collapse_beeps(candidates: list[BeepCandidate], window: float) -> list[BeepCandidate]:
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


def _collapse_gunshots(candidates: list[GunshotCandidate], window: float) -> list[GunshotCandidate]:
    if not candidates:
        return candidates
    candidates.sort(key=lambda c: c.timestamp)
    merged: list[GunshotCandidate] = [candidates[0]]
    for c in candidates[1:]:
        if c.timestamp - merged[-1].timestamp < window:
            if c.peak_amplitude > merged[-1].peak_amplitude:
                merged[-1] = c
        else:
            merged.append(c)
    return merged
