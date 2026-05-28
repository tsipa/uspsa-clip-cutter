"""Tests for beep_detect module — synthetic WAV signals."""

import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from video_stage_cutter.beep_detect import detect_beeps, detect_gunshots


def _make_wav(samples: np.ndarray, sample_rate: int = 16000) -> Path:
    """Write a temporary WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(tmp.name, sample_rate, samples.astype(np.int16))
    return Path(tmp.name)


def _sine(freq: float, duration: float, sr: int = 16000, amplitude: float = 10000) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestDetectBeeps:
    def test_finds_sine_beep_at_known_time(self) -> None:
        sr = 16000
        silence = np.zeros(sr * 3, dtype=np.float32)  # 3s silence
        beep = _sine(3500, 0.15, sr, amplitude=20000)  # 150ms beep at 3500 Hz
        more_silence = np.zeros(sr * 2, dtype=np.float32)  # 2s silence

        audio = np.concatenate([silence, beep, more_silence])
        wav_path = _make_wav(audio, sr)

        try:
            candidates = detect_beeps(wav_path, search_start=2.0, search_end=5.0)
            assert len(candidates) >= 1
            best = max(candidates, key=lambda c: c.energy)
            assert abs(best.timestamp - 3.0) < 0.15, f"Beep at {best.timestamp}, expected ~3.0"
        finally:
            wav_path.unlink(missing_ok=True)

    def test_no_beep_in_silence(self) -> None:
        sr = 16000
        silence = np.zeros(sr * 3, dtype=np.float32)
        wav_path = _make_wav(silence, sr)

        try:
            candidates = detect_beeps(wav_path, search_start=0.0, search_end=3.0)
            assert candidates == []
        finally:
            wav_path.unlink(missing_ok=True)

    def test_low_freq_not_detected_as_beep(self) -> None:
        sr = 16000
        low_tone = _sine(200, 3.0, sr, amplitude=20000)  # 200 Hz, not in beep band
        wav_path = _make_wav(low_tone, sr)

        try:
            candidates = detect_beeps(wav_path, search_start=0.0, search_end=3.0)
            assert candidates == []
        finally:
            wav_path.unlink(missing_ok=True)


    def test_strongest_energy_selected_among_multiple(self) -> None:
        sr = 16000
        # two beeps: weak at 1.0s, strong at 2.0s
        silence1 = np.zeros(sr * 1, dtype=np.float32)
        weak_beep = _sine(3500, 0.1, sr, amplitude=5000)
        gap = np.zeros(int(sr * 0.9), dtype=np.float32)
        strong_beep = _sine(3500, 0.15, sr, amplitude=25000)
        silence2 = np.zeros(sr * 1, dtype=np.float32)

        audio = np.concatenate([silence1, weak_beep, gap, strong_beep, silence2])
        wav_path = _make_wav(audio, sr)

        try:
            candidates = detect_beeps(wav_path, search_start=0.5, search_end=3.5)
            assert len(candidates) >= 2, f"Expected >=2 candidates, got {len(candidates)}"
            best_energy = max(candidates, key=lambda c: c.energy)
            best_composite = max(candidates, key=lambda c: (c.confidence, c.energy))
            # the strong beep should be at ~2.0s
            assert abs(best_energy.timestamp - 2.0) < 0.2, f"Best energy at {best_energy.timestamp}, expected ~2.0"
            # composite key should also pick the strong one
            assert abs(best_composite.timestamp - 2.0) < 0.2
        finally:
            wav_path.unlink(missing_ok=True)


class TestDetectGunshots:
    def test_finds_amplitude_spike(self) -> None:
        sr = 16000
        quiet = np.random.normal(0, 100, sr * 3).astype(np.float32)
        # inject a loud spike at ~1.5s
        spike_start = int(1.5 * sr)
        quiet[spike_start:spike_start + 200] = 30000.0
        wav_path = _make_wav(quiet, sr)

        try:
            candidates = detect_gunshots(wav_path)
            assert len(candidates) >= 1
            best = max(candidates, key=lambda c: c.peak_amplitude)
            assert abs(best.timestamp - 1.5) < 0.2
        finally:
            wav_path.unlink(missing_ok=True)

    def test_no_gunshots_in_silence(self) -> None:
        sr = 16000
        silence = np.zeros(sr * 2, dtype=np.float32)
        wav_path = _make_wav(silence, sr)

        try:
            candidates = detect_gunshots(wav_path)
            assert candidates == []
        finally:
            wav_path.unlink(missing_ok=True)


# --- placeholder for integration tests with real video files ---

class TestWithRealVideos:
    """Placeholder: drop real .mp4 files into tests/fixtures/ and add tests here."""

    def test_placeholder_real_video_detection(self) -> None:
        fixtures = Path(__file__).parent / "fixtures"
        if not fixtures.exists():
            return  # skip: no fixtures directory
        videos = list(fixtures.glob("*.mp4")) + list(fixtures.glob("*.mov"))
        if not videos:
            return  # skip: no video files
        # TODO: process each video and assert detection results
        assert True
