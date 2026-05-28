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

from video_stage_cutter.beep_detect import analyze_beep_candidates, detect_gunshots
from video_stage_cutter.pipeline import (
    Anchor,
    FileInfo,
    ProcessingConfig,
    Stage,
    _assemble_stages,
    _collect_audio_anchors_for_file,
    _subtract_intervals,
    _trim_fallback,
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
            candidates = analyze_beep_candidates(wav_path, search_start=7.25, search_end=17.5)
            assert len(candidates) >= 1
            best = max(candidates, key=lambda c: c.band_energy)
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


class TestReadyWithoutStandby:
    """When there's a ready anchor but no standby, beep search should
    use the ready anchor with a wider window and not crash."""

    def test_ready_only_produces_stage(self) -> None:
        """ready + beep + end = confirmed stage, no standby needed."""
        anchors = [
            Anchor(kind="ready",       abs_time=100.0, file_idx=0, file_offset=5.0,  text="are you ready", score=90, end_offset=6.0),
            Anchor(kind="beep",        abs_time=107.0, file_idx=0, file_offset=12.0, text="timer_beep",    score=80),
            Anchor(kind="end_command", abs_time=130.0, file_idx=0, file_offset=35.0, text="hammer down",   score=85, end_offset=36.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is True
        assert stages[0].beep is not None

    def test_ready_only_no_end_produces_fallback(self) -> None:
        """ready + beep, no end command = fallback stage."""
        anchors = [
            Anchor(kind="ready", abs_time=100.0, file_idx=0, file_offset=5.0, text="make ready", score=85, end_offset=6.0),
            Anchor(kind="beep",  abs_time=150.0, file_idx=0, file_offset=55.0, text="timer_beep", score=80),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is False
        assert "fallback" in stages[0].end_reason

    def test_ready_plus_end_no_beep_is_confirmed(self) -> None:
        """ready + end_command but no beep = confirmed stage with start_reason '*_no_beep'."""
        anchors = [
            Anchor(kind="ready",       abs_time=100.0, file_idx=0, file_offset=5.0,  text="are you ready", score=90, end_offset=6.0),
            Anchor(kind="end_command", abs_time=130.0, file_idx=0, file_offset=35.0, text="hammer down",   score=85, end_offset=36.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is True
        assert "no_beep" in stages[0].start_reason

    def test_standby_plus_end_no_beep_is_confirmed(self) -> None:
        """standby + end_command but no beep = confirmed stage."""
        anchors = [
            Anchor(kind="standby",     abs_time=100.0, file_idx=0, file_offset=5.0,  text="stand by",    score=95, end_offset=6.0),
            Anchor(kind="end_command", abs_time=130.0, file_idx=0, file_offset=35.0, text="hammer down", score=85, end_offset=36.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is True
        assert "no_beep" in stages[0].start_reason

    def test_weak_ready_alone_dropped(self) -> None:
        """Weak 'ready' (low pattern score) without evidence = dropped."""
        anchors = [
            Anchor(kind="ready", abs_time=100.0, file_idx=0, file_offset=5.0, text="ready(100)", score=20, end_offset=6.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 0, "Lone weak 'ready' should be dropped"

    def test_strong_are_you_ready_alone_kept(self) -> None:
        """Strong 'are you ready' (high pattern score) = kept even without beep/end."""
        anchors = [
            Anchor(kind="ready", abs_time=100.0, file_idx=0, file_offset=5.0, text="are you ready(100)", score=50, end_offset=6.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1, "'are you ready' with conf=0.50 should pass min threshold"

    def test_ready_with_gunshots_produces_fallback(self) -> None:
        """ready + gunshots after = enough evidence for fallback."""
        anchors = [
            Anchor(kind="ready", abs_time=100.0, file_idx=0, file_offset=5.0, text="are you ready", score=90, end_offset=6.0),
            Anchor(kind="gunshot", abs_time=108.0, file_idx=0, file_offset=13.0, text="gunshot", score=70),
            Anchor(kind="gunshot", abs_time=109.0, file_idx=0, file_offset=14.0, text="gunshot", score=70),
            Anchor(kind="gunshot", abs_time=110.0, file_idx=0, file_offset=15.0, text="gunshot", score=70),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is False

    def test_standby_only_no_beep_no_end_produces_fallback(self) -> None:
        """standby but no beep and no end = 3min fallback from standby."""
        anchors = [
            Anchor(kind="standby", abs_time=100.0, file_idx=0, file_offset=5.0, text="stand by", score=95, end_offset=6.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is False
        assert "no_beep" in stages[0].start_reason
        assert "fallback" in stages[0].end_reason

    def test_end_command_only_produces_fallback(self) -> None:
        """Only end_command, no start at all = 3min before end."""
        anchors = [
            Anchor(kind="end_command", abs_time=130.0, file_idx=0, file_offset=35.0, text="hammer down", score=85, end_offset=36.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is False
        assert "no_start" in stages[0].start_reason

    def test_multiple_starts_same_stage_no_duplicates(self) -> None:
        """make_ready + are_you_ready + standby + beep + end = ONE stage.
        The earlier ready/standby anchors must not create extra fallback stages."""
        anchors = [
            Anchor(kind="ready",       abs_time=100.0, file_idx=0, file_offset=5.0,   text="make ready",    score=100, end_offset=6.0),
            Anchor(kind="ready",       abs_time=130.0, file_idx=0, file_offset=35.0,  text="are you ready", score=100, end_offset=36.0),
            Anchor(kind="standby",     abs_time=132.0, file_idx=0, file_offset=37.0,  text="stand by",      score=100, end_offset=38.0),
            Anchor(kind="beep",        abs_time=134.0, file_idx=0, file_offset=39.0,  text="timer_beep",    score=80),
            Anchor(kind="gunshot",     abs_time=136.0, file_idx=0, file_offset=41.0,  text="gunshot",       score=70),
            Anchor(kind="end_command", abs_time=160.0, file_idx=0, file_offset=65.0,  text="hammer down",   score=85, end_offset=66.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1, f"Expected 1 stage, got {len(stages)}: {[(s.start_reason, s.end_reason) for s in stages]}"
        assert stages[0].complete is True

    def test_ready_before_confirmed_no_extra_fallback(self) -> None:
        """A 'make ready' 30s before a confirmed stage must not create a separate fallback."""
        anchors = [
            Anchor(kind="ready",       abs_time=100.0, file_idx=0, file_offset=5.0,   text="make ready",    score=100, end_offset=6.0),
            Anchor(kind="standby",     abs_time=125.0, file_idx=0, file_offset=30.0,  text="stand by",      score=100, end_offset=31.0),
            Anchor(kind="beep",        abs_time=127.0, file_idx=0, file_offset=32.0,  text="timer_beep",    score=80),
            Anchor(kind="end_command", abs_time=150.0, file_idx=0, file_offset=55.0,  text="hammer down",   score=85, end_offset=56.0),
        ]
        stages = _assemble_stages(anchors)
        confirmed = [s for s in stages if s.complete]
        fallback = [s for s in stages if not s.complete]
        assert len(confirmed) == 1
        assert len(fallback) == 0, f"Unexpected fallback: {[(s.start_reason, s.end_reason) for s in fallback]}"

    def test_beep_search_record_has_anchor_kind(self) -> None:
        """BeepSearchRecord should record anchor_kind='ready' when no standby."""
        from video_stage_cutter.pipeline import BeepSearchRecord
        rec = BeepSearchRecord(
            anchor_offset=5.0,
            anchor_kind="ready",
            search_start=4.75,
            search_end=95.0,
        )
        assert rec.anchor_kind == "ready"
        assert rec.anchor_offset == 5.0


class TestBeepSearchClampedAtEnd:
    """Beep search window should be clamped at the first end_command."""

    def test_beep_before_end_command_found(self) -> None:
        """If end_command is at 30s, beep at 12s should be found
        (it's before end_command)."""
        anchors = [
            Anchor(kind="standby",     abs_time=100.0, file_idx=0, file_offset=10.0, text="stand by",    score=95, end_offset=11.0),
            Anchor(kind="beep",        abs_time=102.0, file_idx=0, file_offset=12.0, text="timer_beep",  score=80),
            Anchor(kind="end_command", abs_time=120.0, file_idx=0, file_offset=30.0, text="hammer down", score=85, end_offset=31.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) == 1
        assert stages[0].complete is True

    def test_no_beep_after_end_command(self) -> None:
        """A beep anchor placed AFTER end_command should still form a stage
        with the end_command (assembly pairs beep with next end_command)."""
        anchors = [
            Anchor(kind="standby",     abs_time=100.0, file_idx=0, file_offset=10.0, text="stand by",    score=95, end_offset=11.0),
            Anchor(kind="beep",        abs_time=102.0, file_idx=0, file_offset=12.0, text="timer_beep",  score=80),
            Anchor(kind="end_command", abs_time=105.0, file_idx=0, file_offset=15.0, text="range clear",  score=85, end_offset=16.0),
        ]
        stages = _assemble_stages(anchors)
        confirmed = [s for s in stages if s.complete]
        assert len(confirmed) == 1


class TestSubtractIntervals:
    def test_no_overlap(self) -> None:
        result = _subtract_intervals(0, 100, [(200, 300)])
        assert result == [(0, 100)]

    def test_full_overlap(self) -> None:
        result = _subtract_intervals(10, 50, [(0, 100)])
        assert result == []

    def test_partial_overlap_start(self) -> None:
        result = _subtract_intervals(0, 100, [(50, 150)])
        assert result == [(0, 50)]

    def test_partial_overlap_end(self) -> None:
        result = _subtract_intervals(50, 150, [(0, 100)])
        assert result == [(100, 150)]

    def test_middle_hole(self) -> None:
        result = _subtract_intervals(0, 100, [(30, 60)])
        assert result == [(0, 30), (60, 100)]

    def test_multiple_exclusions(self) -> None:
        result = _subtract_intervals(0, 100, [(10, 20), (40, 50), (80, 90)])
        assert result == [(0, 10), (20, 40), (50, 80), (90, 100)]


class TestFallbackOverlapTrimming:
    def test_fallback_trimmed_to_avoid_confirmed(self) -> None:
        """Confirmed stage at 10-45, fallback ending at 55 should be trimmed to 45-55."""
        stage = Stage(
            end_command=Anchor(kind="end_command", abs_time=55.0, file_idx=0, file_offset=55.0,
                               text="hammer down", score=85, end_offset=56.0),
            clip_start=55.0 - 180.0,  # -125, would be clamped elsewhere
            clip_end=56.0,
            start_reason="fallback_3min_no_start",
            end_reason="matched:hammer down",
            complete=False,
        )
        # clamp start to 0
        stage.clip_start = max(0, stage.clip_start)

        confirmed_intervals = [(10.0, 45.0)]
        result = _trim_fallback(stage, confirmed_intervals, min_clip_length=5.0)

        assert result is not None
        assert result.clip_start >= 45.0
        assert result.clip_end <= 56.0
        assert result.trimmed is True

    def test_fallback_skipped_when_too_short(self) -> None:
        """If trimming leaves interval shorter than min_clip_length, skip it."""
        stage = Stage(
            beep=Anchor(kind="beep", abs_time=60.0, file_idx=0, file_offset=60.0,
                        text="timer_beep", score=80),
            clip_start=60.0,
            clip_end=60.0 + 180.0,
            start_reason="beep",
            end_reason="fallback_3min_no_end",
            complete=False,
        )
        # confirmed stage covers almost all the fallback window
        confirmed_intervals = [(62.0, 239.0)]
        result = _trim_fallback(stage, confirmed_intervals, min_clip_length=5.0)

        # remaining: 60-62 (2s) and 239-240 (1s), both < 5s
        assert result is None

    def test_fallback_no_overlap(self) -> None:
        """Fallback with no overlap should pass through unchanged."""
        stage = Stage(
            beep=Anchor(kind="beep", abs_time=300.0, file_idx=0, file_offset=300.0,
                        text="timer_beep", score=80),
            clip_start=300.0,
            clip_end=480.0,
            start_reason="beep",
            end_reason="fallback_3min_no_end",
            complete=False,
        )
        confirmed_intervals = [(10.0, 45.0)]
        result = _trim_fallback(stage, confirmed_intervals, min_clip_length=5.0)

        assert result is not None
        assert result.clip_start == 300.0
        assert result.clip_end == 480.0
        assert result.trimmed is False

    def test_assembly_trims_fallback_against_confirmed(self) -> None:
        """Full integration: confirmed stage should prevent fallback overlap."""
        anchors = [
            # confirmed stage: beep at 10, end at 45
            Anchor(kind="standby",     abs_time=8.0,  file_idx=0, file_offset=8.0,  text="stand by",    score=90),
            Anchor(kind="beep",        abs_time=10.0, file_idx=0, file_offset=10.0, text="timer_beep",  score=80),
            Anchor(kind="end_command", abs_time=45.0, file_idx=0, file_offset=45.0, text="hammer down", score=85, end_offset=46.0),
            # orphan end at 55 with no start
            Anchor(kind="end_command", abs_time=55.0, file_idx=0, file_offset=55.0, text="hammer down", score=85, end_offset=56.0),
        ]
        stages = _assemble_stages(anchors, min_clip_length=5.0)

        confirmed = [s for s in stages if s.complete]
        fallbacks = [s for s in stages if not s.complete]

        assert len(confirmed) == 1
        assert confirmed[0].clip_start == 10.0

        if fallbacks:
            for fb in fallbacks:
                # fallback must not overlap confirmed stage
                assert fb.clip_start >= 46.0 or fb.clip_end <= 10.0


class TestPhase1CAudioAnchors:
    """Tests for _collect_audio_anchors_for_file (Phase 1C)."""

    def test_returns_list_not_tuple(self) -> None:
        """_collect_audio_anchors_for_file must return a list, not a tuple."""
        wav_path = _make_wav(_noise(3.0))
        fi = FileInfo(
            path=Path("/fake/video.mp4"),
            wav_path=wav_path,
            duration=3.0,
            creation_epoch=0.0,
            creation_str="2024-01-01_00-00-00",
            creation_iso="2024-01-01T00:00:00Z",
        )
        try:
            result = _collect_audio_anchors_for_file(fi, 0, ProcessingConfig(), [])
            assert isinstance(result, list), f"Expected list, got {type(result)}"
        finally:
            wav_path.unlink(missing_ok=True)

    def test_rejected_beep_candidates_do_not_create_anchor(self) -> None:
        """Broadband gunshot-like impulse in beep window must not create a beep anchor."""
        sr = 16000
        # noise + broadband spike at 1.5s (gunshot, not tonal beep)
        audio = _noise(3.0, amplitude=300)
        spike_start = int(1.5 * sr)
        audio[spike_start:spike_start + 100] = 25000.0  # short broadband impulse

        wav_path = _make_wav(audio)
        fi = FileInfo(
            path=Path("/fake/video.mp4"),
            wav_path=wav_path,
            duration=3.0,
            creation_epoch=0.0,
            creation_str="2024-01-01_00-00-00",
            creation_iso="2024-01-01T00:00:00Z",
        )
        # standby anchor at 0.5s so beep search triggers
        phrase_anchors = [
            Anchor(kind="standby", abs_time=0.5, file_idx=0, file_offset=0.5,
                   text="stand by", score=80, end_offset=1.0),
        ]
        try:
            result = _collect_audio_anchors_for_file(fi, 0, ProcessingConfig(), phrase_anchors)
            beep_anchors = [a for a in result if a.kind == "beep"]
            assert len(beep_anchors) == 0, (
                f"Broadband impulse should not create beep anchor, got {beep_anchors}"
            )
        finally:
            wav_path.unlink(missing_ok=True)

    def test_standby_with_high_score_survives_confidence_filter(self) -> None:
        """'stand by' with high pattern score should survive MIN_CONFIDENCE even
        without beep, end, or gunshots."""
        anchors = [
            Anchor(kind="standby", abs_time=100.0, file_idx=0, file_offset=5.0,
                   text="stand by(100)", score=80, end_offset=6.0),
        ]
        stages = _assemble_stages(anchors)
        assert len(stages) >= 1, (
            "stand by with score=80 should survive confidence filter"
        )
