"""Audio event detection: timer beeps and gunshots.

Beep detection uses multiple spectral features to distinguish a timer beep
(pure tone ~3000-4000 Hz, 80-500ms) from gunshots (broadband impulse <50ms)
and background noise. All features are logged per candidate so thresholds
can be tuned from real data.
"""

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
    band_energy: float
    tonality: float         # peak_bin / sum_bins in beep band — pure tone ~0.3+
    broadband_ratio: float  # energy_beep_band / energy_wide_band — beep ~0.3+, gunshot ~0.05
    spectral_flatness: float  # geometric_mean / arithmetic_mean — noise ~1.0, tone ~0.0
    duration_ms: float
    neighbors_1s: int       # how many other candidates within ±1s (series = gunshots)
    composite_score: float = 0.0
    reject_reason: str = ""
    accepted: bool = False


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


def analyze_beep_candidates(
    wav_path: Path,
    search_start: float,
    search_end: float,
    freq_low: float = 2500.0,
    freq_high: float = 5000.0,
    wide_freq_low: float = 300.0,
    wide_freq_high: float = 8000.0,
    collapse_window: float = 0.3,
    min_tonality: float = 0.15,
    min_broadband_ratio: float = 0.10,
    max_spectral_flatness: float = 0.8,
    min_duration_ms: float = 80.0,
    max_duration_ms: float = 500.0,
    max_neighbors_1s: int = 3,
) -> list[BeepCandidate]:
    """Detect high-frequency tonal beeps.

    All thresholds are configurable. Every candidate is logged with all
    features regardless of accept/reject so thresholds can be tuned from
    real data.
    """
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

    beep_band = (freqs >= freq_low) & (freqs <= freq_high)
    wide_band = (freqs >= wide_freq_low) & (freqs <= wide_freq_high)

    if not beep_band.any():
        log.warning("No frequency bins in %.0f-%.0f Hz range", freq_low, freq_high)
        return []

    beep_energy = Sxx[beep_band, :].mean(axis=0)
    wide_energy = Sxx[wide_band, :].mean(axis=0) if wide_band.any() else beep_energy

    if beep_energy.max() == 0:
        return []

    median_energy = float(np.median(beep_energy))
    std_energy = float(np.std(beep_energy))
    threshold = median_energy + 3.0 * std_energy

    dt = float(times[1] - times[0]) if len(times) > 1 else 0.032

    # find contiguous runs above threshold
    above = beep_energy >= threshold
    raw_runs: list[tuple[int, int, int]] = []  # (run_start, run_end, peak_idx)
    i = 0
    while i < len(above):
        if above[i]:
            run_start = i
            peak_idx = i
            peak_val = beep_energy[i]
            while i < len(above) and above[i]:
                if beep_energy[i] > peak_val:
                    peak_val = beep_energy[i]
                    peak_idx = i
                i += 1
            raw_runs.append((run_start, i, peak_idx))
        else:
            i += 1

    # compute features for each run
    candidates: list[BeepCandidate] = []
    for run_start, run_end, peak_idx in raw_runs:
        abs_time = search_start + float(times[peak_idx])
        duration_ms = (run_end - run_start) * dt * 1000.0

        # tonality: peak FFT bin / sum of bins in beep band at peak frame
        beep_spectrum = Sxx[beep_band, peak_idx]
        tonality = float(beep_spectrum.max() / (beep_spectrum.sum() + 1e-9))

        # broadband ratio: beep band energy / wide band energy
        bb_ratio = float(beep_energy[peak_idx] / (wide_energy[peak_idx] + 1e-9))

        # spectral flatness at peak frame (full spectrum)
        full_spectrum = Sxx[:, peak_idx]
        full_spectrum_pos = full_spectrum[full_spectrum > 0]
        if len(full_spectrum_pos) > 0:
            geo_mean = float(np.exp(np.mean(np.log(full_spectrum_pos + 1e-20))))
            arith_mean = float(np.mean(full_spectrum_pos))
            flatness = geo_mean / (arith_mean + 1e-9)
        else:
            flatness = 1.0

        candidates.append(BeepCandidate(
            timestamp=abs_time,
            band_energy=float(beep_energy[peak_idx]),
            tonality=tonality,
            broadband_ratio=bb_ratio,
            spectral_flatness=flatness,
            duration_ms=duration_ms,
            neighbors_1s=0,
        ))

    # count neighbors within ±1s (series detection)
    for i, c in enumerate(candidates):
        count = 0
        for j, other in enumerate(candidates):
            if i != j and abs(c.timestamp - other.timestamp) <= 1.0:
                count += 1
        c.neighbors_1s = count

    # compute composite score for ranking
    max_energy = max((c.band_energy for c in candidates), default=1.0)
    for c in candidates:
        norm_energy = c.band_energy / (max_energy + 1e-9)
        dur_score = min(1.0, c.duration_ms / 200.0) if c.duration_ms <= 500 else 0.5
        c.composite_score = (
            0.45 * norm_energy
            + 0.35 * c.tonality
            + 0.15 * c.broadband_ratio
            + 0.05 * dur_score
        )

    # log ALL candidates with features
    log.info(
        "Beep detection: searched %.2f-%.2fs, threshold=%.1f, %d raw candidates",
        search_start, search_end, threshold, len(candidates),
    )
    for c in candidates:
        log.info(
            "  RAW  t=%.3fs energy=%.1f tonality=%.3f bb_ratio=%.3f flatness=%.3f "
            "duration=%.0fms neighbors=%d score=%.3f",
            c.timestamp, c.band_energy, c.tonality, c.broadband_ratio,
            c.spectral_flatness, c.duration_ms, c.neighbors_1s, c.composite_score,
        )

    # apply filters — log reject reason for each
    for c in candidates:
        reasons = []
        if c.duration_ms < min_duration_ms:
            reasons.append(f"too_short({c.duration_ms:.0f}ms<{min_duration_ms:.0f}ms)")
        if c.duration_ms > max_duration_ms:
            reasons.append(f"too_long({c.duration_ms:.0f}ms>{max_duration_ms:.0f}ms)")
        if c.tonality < min_tonality:
            reasons.append(f"low_tonality({c.tonality:.3f}<{min_tonality})")
        if c.broadband_ratio < min_broadband_ratio:
            reasons.append(f"broadband({c.broadband_ratio:.3f}<{min_broadband_ratio})")
        if c.spectral_flatness > max_spectral_flatness:
            reasons.append(f"flat_spectrum({c.spectral_flatness:.3f}>{max_spectral_flatness})")
        if c.neighbors_1s > max_neighbors_1s:
            reasons.append(f"series({c.neighbors_1s}neighbors>{max_neighbors_1s})")

        if reasons:
            c.reject_reason = ",".join(reasons)
            c.accepted = False
        else:
            c.accepted = True

    for c in candidates:
        if c.accepted:
            log.info("  ACCEPT t=%.3fs energy=%.1f tonality=%.3f bb_ratio=%.3f duration=%.0fms",
                     c.timestamp, c.band_energy, c.tonality, c.broadband_ratio, c.duration_ms)
        else:
            log.info("  REJECT t=%.3fs: %s", c.timestamp, c.reject_reason)

    accepted_raw = [c for c in candidates if c.accepted]
    accepted_collapsed = _collapse_beeps(accepted_raw, collapse_window)
    collapsed_ids = {id(c) for c in accepted_collapsed}

    for c in candidates:
        if c.accepted and id(c) not in collapsed_ids:
            c.accepted = False
            c.reject_reason = "collapsed_duplicate"

    num_accepted = sum(1 for c in candidates if c.accepted)
    log.info("  Result: %d accepted / %d rejected / %d total",
             num_accepted, len(candidates) - num_accepted, len(candidates))
    if num_accepted == 0:
        log.warning("  No beep survived filtering in window %.2f-%.2fs", search_start, search_end)

    return candidates


def detect_gunshots(
    wav_path: Path,
    search_start: float = 0.0,
    search_end: float | None = None,
    collapse_window: float = 0.15,
) -> list[GunshotCandidate]:
    """Detect loud transient spikes (gunshots) in audio."""
    sample_rate, data = load_wav(wav_path)

    if search_end is None:
        search_end = len(data) / sample_rate

    start_sample = max(0, int(search_start * sample_rate))
    end_sample = min(len(data), int(search_end * sample_rate))

    if end_sample <= start_sample:
        return []

    segment = np.abs(data[start_sample:end_sample])

    window_samples = int(0.01 * sample_rate)
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
            if c.band_energy > merged[-1].band_energy:
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
