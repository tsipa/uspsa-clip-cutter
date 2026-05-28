"""Tests using real Whisper transcripts from DJI action camera footage.

Pattern-based: tests verify that keyword patterns are detected,
not specific matched_phrase values.
"""

from __future__ import annotations

import json
from pathlib import Path

from video_stage_cutter.phrase_detect import detect_phrases
from video_stage_cutter.pipeline import (
    Anchor,
    _assemble_stages,
)
from video_stage_cutter.transcribe import TranscriptSegment, WordInfo

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcripts"


def _load(name: str) -> list[TranscriptSegment]:
    path = FIXTURES_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: list[TranscriptSegment] = []
    for seg in data["segments"]:
        words = [
            WordInfo(start=w["start"], end=w["end"], word=w["word"], probability=w["probability"])
            for w in seg.get("words", [])
        ]
        segments.append(TranscriptSegment(start=seg["start"], end=seg["end"], text=seg["text"], words=words))
    return segments


class TestDJI0287:
    """Whisper heard nothing useful. No stage."""

    def test_no_start_no_end(self) -> None:
        segments = _load("real_dji_0287")
        starts, ends = detect_phrases(segments)
        assert len(starts) == 0
        assert len(ends) == 0


class TestDJI0288:
    """Contains stand by, hammer down, range is clear."""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0288")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1

    def test_finds_end(self) -> None:
        segments = _load("real_dji_0288")
        starts, ends = detect_phrases(segments)
        assert len(ends) >= 1


class TestDJI0291:
    """Contains stand by and hammer down."""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0291")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1

    def test_finds_end(self) -> None:
        segments = _load("real_dji_0291")
        starts, ends = detect_phrases(segments)
        assert len(ends) >= 1

    def test_produces_stage_with_beep(self) -> None:
        segments = _load("real_dji_0291")
        starts, ends = detect_phrases(segments)
        anchors = _build_anchors(starts, ends)
        standby_a = [a for a in anchors if a.kind == "standby"]
        if standby_a:
            sb = standby_a[-1]
            anchors.append(Anchor(
                kind="beep", abs_time=sb.end_offset + 1.0, file_idx=0,
                file_offset=sb.end_offset + 1.0, text="timer_beep", score=80,
            ))
        stages = _assemble_stages(anchors, min_clip_length=5.0)
        assert len(stages) >= 1


class TestDJI0292:
    """Contains make ready, are you ready, stand by, unload, hammer down."""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0292")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1

    def test_finds_end(self) -> None:
        segments = _load("real_dji_0292")
        starts, ends = detect_phrases(segments)
        assert len(ends) >= 1


class TestDJI0294:
    """Contains are you ready, stand by, unload, hammer down."""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0294")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1

    def test_finds_end(self) -> None:
        segments = _load("real_dji_0294")
        starts, ends = detect_phrases(segments)
        assert len(ends) >= 1


class TestDJI0295:
    """Contains stand by, hammer down and holster, range is clear."""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0295")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1

    def test_finds_end(self) -> None:
        segments = _load("real_dji_0295")
        starts, ends = detect_phrases(segments)
        assert len(ends) >= 1

    def test_no_duplicate_end_patterns(self) -> None:
        """Multiple end keywords from same moment should group into one pattern."""
        segments = _load("real_dji_0295")
        starts, ends = detect_phrases(segments)
        # should not have many separate end patterns from the same RO command
        assert len(ends) <= 3


class TestDJI0297_001_Small:
    """small model: only heard 'Are you ready? Stand by.'"""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0297_001_small")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1


class TestDJI0297_001_Large:
    """large-v3 model: heard hammer down and holster + stand by."""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0297_001_large")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1

    def test_finds_end(self) -> None:
        segments = _load("real_dji_0297_001_large")
        starts, ends = detect_phrases(segments)
        assert len(ends) >= 1


class TestDJI0298_Small:
    """small model: merged segments."""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0298_small")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1


class TestDJI0298_Large:
    """large-v3 model."""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0298_large")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1


def _build_anchors(starts, ends, epoch=0.0):
    anchors: list[Anchor] = []
    for m in starts:
        kind = "standby" if "stand" in m.matched_phrase.lower() else "ready"
        anchors.append(Anchor(
            kind=kind, abs_time=epoch + m.start, file_idx=0,
            file_offset=m.start, text=m.text, score=m.score, end_offset=m.end,
        ))
    for m in ends:
        anchors.append(Anchor(
            kind="end_command", abs_time=epoch + m.start, file_idx=0,
            file_offset=m.start, text=m.text, score=m.score, end_offset=m.end,
        ))
    return anchors
