"""End-to-end pipeline test with synthetic audio and fake transcript.

We can't synthesize speech, so we:
1. Build a WAV with the right audio events (beep, gunshots, silence)
2. Mock faster-whisper to return a scripted transcript
3. Run the full anchor collection + stage assembly
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.io import wavfile

from video_stage_cutter.beep_detect import detect_beeps, detect_gunshots
from video_stage_cutter.pipeline import (
    Anchor,
    ProcessingConfig,
    Stage,
    _assemble_stages,
)


SR = 16000


def _sine(freq: float, duration: float, amplitude: float = 20000) -> np.ndarray:
    t = np.arange(int(SR * duration)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration: float) -> np.ndarray:
    return np.zeros(int(SR * duration), dtype=np.float32)


def _noise(duration: float, amplitude: float = 300) -> np.ndarray:
    return (np.random.default_rng(42).normal(0, amplitude, int(SR * duration))).astype(np.float32)


def _spike(duration: float = 0.01, amplitude: float = 30000) -> np.ndarray:
    return np.full(int(SR * duration), amplitude, dtype=np.float32)


def _make_wav(samples: np.ndarray) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(tmp.name, SR, samples.astype(np.int16))
    return Path(tmp.name)


def build_stage_audio() -> tuple[Path, dict]:
    """Build a synthetic stage WAV:

    Timeline (seconds):
      0-5     background noise
      5.0     "are you ready" would be here (transcript only)
      7.0     "stand by" would be here (transcript only)
      8.5     BEEP (3500 Hz, 150ms)
      10-12   gunshot spikes (3 shots)
      15-17   gunshot spikes (3 more shots)
      25.0    "if clear hammer down and holster" (transcript only)
      25-30   background noise / silence

    Returns (wav_path, expected_times).
    """
    parts = []

    # 0-5: background noise
    parts.append(_noise(5.0))

    # 5-7: noise (speech would be here)
    parts.append(_noise(2.0))

    # 7-8.5: noise (after "stand by", before beep)
    parts.append(_noise(1.5))

    # 8.5: beep 150ms
    parts.append(_sine(3500, 0.15))

    # 8.65-10: noise
    parts.append(_noise(1.35))

    # 10-12: 3 gunshots, ~0.7s apart
    for _ in range(3):
        parts.append(_spike())
        parts.append(_noise(0.69))

    # 12-15: noise
    parts.append(_noise(2.93))

    # 15-17: 3 more gunshots
    for _ in range(3):
        parts.append(_spike())
        parts.append(_noise(0.69))

    # 17-25: noise
    parts.append(_noise(7.93))

    # 25-30: noise (end command would be here)
    parts.append(_noise(5.0))

    audio = np.concatenate(parts)
    wav_path = _make_wav(audio)

    return wav_path, {
        "beep_time": 8.5,
        "gunshot_start": 10.0,
        "gunshot_end": 17.0,
        "duration": len(audio) / SR,
    }


class TestSyntheticBeepInStage:
    def test_beep_found_at_correct_time(self) -> None:
        wav_path, expected = build_stage_audio()
        try:
            # search around where "stand by" ends (~7.5s)
            candidates = detect_beeps(wav_path, search_start=7.25, search_end=17.5)
            assert len(candidates) >= 1
            best = max(candidates, key=lambda c: c.energy)
            assert abs(best.timestamp - expected["beep_time"]) < 0.2, (
                f"Beep at {best.timestamp}, expected ~{expected['beep_time']}"
            )
        finally:
            wav_path.unlink(missing_ok=True)

    def test_gunshots_found(self) -> None:
        wav_path, expected = build_stage_audio()
        try:
            candidates = detect_gunshots(wav_path)
            assert len(candidates) >= 4, f"Expected >=4 gunshots, got {len(candidates)}"
            times = [c.timestamp for c in candidates]
            in_range = [t for t in times if expected["gunshot_start"] - 0.5 <= t <= expected["gunshot_end"] + 0.5]
            assert len(in_range) >= 4, f"Only {len(in_range)} gunshots in expected range"
        finally:
            wav_path.unlink(missing_ok=True)


class TestAssembleStages:
    """Test stage assembly with hand-crafted anchors (no audio needed)."""

    def test_single_complete_stage(self) -> None:
        anchors = [
            Anchor(kind="ready",       abs_time=100.0, file_idx=0, file_offset=5.0,  text="are you ready", score=90),
            Anchor(kind="standby",     abs_time=102.0, file_idx=0, file_offset=7.0,  text="stand by",      score=95),
            Anchor(kind="beep",        abs_time=103.5, file_idx=0, file_offset=8.5,  text="timer_beep",    score=80),
            Anchor(kind="gunshot",     abs_time=105.0, file_idx=0, file_offset=10.0, text="gunshot",       score=70),
            Anchor(kind="gunshot",     abs_time=106.0, file_idx=0, file_offset=11.0, text="gunshot",       score=70),
            Anchor(kind="end_command", abs_time=120.0, file_idx=0, file_offset=25.0, text="hammer down",   score=85, end_offset=26.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        s = stages[0]
        assert s.complete is True
        assert s.beep is not None
        assert s.end_command is not None
        assert s.ready is not None
        assert s.standby is not None
        assert len(s.gunshots) >= 2

    def test_two_stages_same_file(self) -> None:
        anchors = [
            # stage 1
            Anchor(kind="standby",     abs_time=100.0, file_idx=0, file_offset=5.0,   text="stand by",      score=90),
            Anchor(kind="beep",        abs_time=101.5, file_idx=0, file_offset=6.5,   text="timer_beep",    score=80),
            Anchor(kind="gunshot",     abs_time=103.0, file_idx=0, file_offset=8.0,   text="gunshot",       score=70),
            Anchor(kind="end_command", abs_time=115.0, file_idx=0, file_offset=20.0,  text="hammer down",   score=85, end_offset=21.0),
            # stage 2
            Anchor(kind="standby",     abs_time=200.0, file_idx=0, file_offset=105.0, text="stand by",      score=90),
            Anchor(kind="beep",        abs_time=201.5, file_idx=0, file_offset=106.5, text="timer_beep",    score=80),
            Anchor(kind="gunshot",     abs_time=204.0, file_idx=0, file_offset=109.0, text="gunshot",       score=70),
            Anchor(kind="end_command", abs_time=220.0, file_idx=0, file_offset=125.0, text="hammer down",   score=85, end_offset=126.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 2
        assert stages[0].complete is True
        assert stages[1].complete is True
        assert stages[0].clip_start < stages[1].clip_start

    def test_cross_file_stage(self) -> None:
        anchors = [
            Anchor(kind="standby",     abs_time=100.0, file_idx=0, file_offset=50.0,  text="stand by",    score=90),
            Anchor(kind="beep",        abs_time=101.5, file_idx=0, file_offset=51.5,  text="timer_beep",  score=80),
            Anchor(kind="gunshot",     abs_time=103.0, file_idx=0, file_offset=53.0,  text="gunshot",     score=70),
            # file boundary here, end_command in next file
            Anchor(kind="end_command", abs_time=115.0, file_idx=1, file_offset=5.0,   text="hammer down", score=85, end_offset=6.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is True
        assert stages[0].beep.file_idx == 0
        assert stages[0].end_command.file_idx == 1

    def test_no_end_command_fallback(self) -> None:
        anchors = [
            Anchor(kind="standby", abs_time=100.0, file_idx=0, file_offset=5.0, text="stand by",   score=90),
            Anchor(kind="beep",    abs_time=101.5, file_idx=0, file_offset=6.5, text="timer_beep", score=80),
            Anchor(kind="gunshot", abs_time=103.0, file_idx=0, file_offset=8.0, text="gunshot",    score=70),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is False
        assert "fallback" in stages[0].end_reason
        assert stages[0].duration <= 181  # ~180s fallback

    def test_orphan_end_command(self) -> None:
        anchors = [
            Anchor(kind="gunshot",     abs_time=103.0, file_idx=0, file_offset=8.0,  text="gunshot",     score=70),
            Anchor(kind="end_command", abs_time=120.0, file_idx=0, file_offset=25.0, text="hammer down", score=85, end_offset=26.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is False
        assert "no_start" in stages[0].start_reason

    def test_no_anchors(self) -> None:
        stages = _assemble_stages([])
        assert stages == []

    def test_gunshots_dont_leak_between_stages(self) -> None:
        anchors = [
            Anchor(kind="beep",        abs_time=100.0, file_idx=0, file_offset=5.0,   text="timer_beep",  score=80),
            Anchor(kind="gunshot",     abs_time=102.0, file_idx=0, file_offset=7.0,   text="gunshot",     score=70),
            Anchor(kind="end_command", abs_time=110.0, file_idx=0, file_offset=15.0,  text="hammer down", score=85, end_offset=16.0),
            # gap, unrelated gunshot
            Anchor(kind="gunshot",     abs_time=150.0, file_idx=0, file_offset=55.0,  text="gunshot",     score=70),
            # stage 2
            Anchor(kind="beep",        abs_time=200.0, file_idx=0, file_offset=105.0, text="timer_beep",  score=80),
            Anchor(kind="gunshot",     abs_time=202.0, file_idx=0, file_offset=107.0, text="gunshot",     score=70),
            Anchor(kind="end_command", abs_time=215.0, file_idx=0, file_offset=120.0, text="hammer down", score=85, end_offset=121.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 2
        # gunshot at 150s should not be in either stage
        stage1_gs_times = [g.abs_time for g in stages[0].gunshots]
        stage2_gs_times = [g.abs_time for g in stages[1].gunshots]
        assert 150.0 not in stage1_gs_times
        assert 150.0 not in stage2_gs_times
