"""Tests using real Whisper transcripts from DJI action camera footage.

These fixtures are actual transcription results from USPSA match videos.
They test that phrase detection handles real-world ASR output correctly.
No video files or Whisper model needed — just the saved transcript JSONs.
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


def _build_anchors(starts, ends, epoch=0.0):
    anchors: list[Anchor] = []
    for m in starts:
        kind = "standby" if "stand by" in m.matched_phrase.lower() else "ready"
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


class TestDJI0287:
    """Whisper heard almost nothing useful — just 'Made up. Made up in alpha.'
    No RO commands detected. Should produce no start and no end."""

    def test_no_start_no_end(self) -> None:
        segments = _load("real_dji_0287")
        starts, ends = detect_phrases(segments)
        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) == 0
        hammer = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(hammer) == 0


class TestDJI0288:
    """Contains: 'Stand by. Unload. Show clear. If clear, hammer down and holster.'
    and 'Range is clear.' Should find both start and end."""

    def test_finds_standby(self) -> None:
        segments = _load("real_dji_0288")
        starts, ends = detect_phrases(segments)
        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) >= 1

    def test_finds_hammer_down(self) -> None:
        segments = _load("real_dji_0288")
        starts, ends = detect_phrases(segments)
        hammer = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(hammer) >= 1

    def test_finds_range_is_clear(self) -> None:
        segments = _load("real_dji_0288")
        starts, ends = detect_phrases(segments)
        ric = [m for m in ends if "range is clear" in m.matched_phrase.lower()]
        assert len(ric) >= 1


class TestDJI0291:
    """Contains: 'Stand by.' at ~3-70s and 'Clear. Hammer down.' at ~70-95s.
    Should find start and end."""

    def test_finds_standby(self) -> None:
        segments = _load("real_dji_0291")
        starts, ends = detect_phrases(segments)
        standby = [m for m in starts if "stand" in m.matched_phrase.lower()]
        assert len(standby) >= 1

    def test_finds_hammer_down(self) -> None:
        segments = _load("real_dji_0291")
        starts, ends = detect_phrases(segments)
        hammer = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(hammer) >= 1

    def test_produces_confirmed_stage_with_beep(self) -> None:
        segments = _load("real_dji_0291")
        starts, ends = detect_phrases(segments)
        anchors = _build_anchors(starts, ends)
        # add synthetic beep after last standby
        standby_a = [a for a in anchors if a.kind == "standby"]
        if standby_a:
            sb = standby_a[-1]
            anchors.append(Anchor(
                kind="beep", abs_time=sb.end_offset + 1.0, file_idx=0,
                file_offset=sb.end_offset + 1.0, text="timer_beep", score=80,
            ))
        stages = _assemble_stages(anchors, min_clip_length=5.0)
        confirmed = [s for s in stages if s.complete]
        assert len(confirmed) >= 1


class TestDJI0292:
    """Contains: 'make ready' at ~140.8s, 'unload so clear' at ~187.9s,
    'Clear, hammer down' at ~203.4s, 'Rage is clear' at ~207.2s.
    Previously failed because 'make ready' wasn't in START_PHRASES."""

    def test_finds_make_ready(self) -> None:
        segments = _load("real_dji_0292")
        starts, ends = detect_phrases(segments)
        make_ready = [m for m in starts if "make ready" in m.matched_phrase.lower()]
        assert len(make_ready) >= 1, (
            f"'make ready' not found. Got starts: {[(m.text, m.matched_phrase, m.score) for m in starts]}"
        )

    def test_finds_end_command(self) -> None:
        """Should find at least one end: 'unload show clear' or 'hammer down' or 'range is clear'."""
        segments = _load("real_dji_0292")
        starts, ends = detect_phrases(segments)
        assert len(ends) >= 1, f"No end commands found at all"


class TestDJI0294:
    """Contains: 'Are you ready?' at ~76.8s, 'Stand by.' at ~79.3s,
    'Range is clear.' at ~98.0s. Previously failed because 'range is clear'
    wasn't in END_PHRASES."""

    def test_finds_are_you_ready(self) -> None:
        segments = _load("real_dji_0294")
        starts, ends = detect_phrases(segments)
        ready = [m for m in starts if "are you ready" in m.matched_phrase.lower()]
        assert len(ready) >= 1

    def test_finds_stand_by(self) -> None:
        segments = _load("real_dji_0294")
        starts, ends = detect_phrases(segments)
        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) >= 1

    def test_finds_range_is_clear(self) -> None:
        segments = _load("real_dji_0294")
        starts, ends = detect_phrases(segments)
        ric = [m for m in ends if "range is clear" in m.matched_phrase.lower()]
        assert len(ric) >= 1

    def test_make_ready_also_found(self) -> None:
        segments = _load("real_dji_0294")
        starts, ends = detect_phrases(segments)
        mr = [m for m in starts if "make ready" in m.matched_phrase.lower()]
        assert len(mr) >= 1, "Whisper heard 'Make ready.' at ~21.5-27.7s"


class TestDJI0295:
    """Contains: 'Stand by. Hammer down and holster.' at ~145.9-188.2s
    and 'Range is clear.' at ~189.0s. Should produce one confirmed stage,
    not multiple duplicates."""

    def test_finds_standby(self) -> None:
        segments = _load("real_dji_0295")
        starts, ends = detect_phrases(segments)
        standby = [m for m in starts if "stand" in m.matched_phrase.lower()]
        assert len(standby) >= 1

    def test_finds_hammer_down(self) -> None:
        segments = _load("real_dji_0295")
        starts, ends = detect_phrases(segments)
        hammer = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(hammer) >= 1

    def test_no_duplicate_end_commands(self) -> None:
        """Multiple end phrase variants from same moment should deduplicate."""
        segments = _load("real_dji_0295")
        starts, ends = detect_phrases(segments)
        # should not have multiple hammer-related matches at the same time
        hammer = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        if len(hammer) > 1:
            for i in range(1, len(hammer)):
                assert abs(hammer[i].start - hammer[0].start) > 3.0, (
                    f"Duplicate hammer matches within 3s: {[(m.start, m.text) for m in hammer]}"
                )


class TestDJI0297_001_Small:
    """small model: only heard 'Are you ready? Stand by.' — no end command."""

    def test_finds_standby(self) -> None:
        segments = _load("real_dji_0297_001_small")
        starts, ends = detect_phrases(segments)
        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) >= 1

    def test_no_hammer_down(self) -> None:
        segments = _load("real_dji_0297_001_small")
        starts, ends = detect_phrases(segments)
        hammer = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(hammer) == 0


class TestDJI0297_001_Large:
    """large-v3 model: heard 'hammer down and holster' + 'Are you ready? Stand by.'"""

    def test_finds_standby(self) -> None:
        segments = _load("real_dji_0297_001_large")
        starts, ends = detect_phrases(segments)
        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) >= 1

    def test_finds_hammer_down(self) -> None:
        segments = _load("real_dji_0297_001_large")
        starts, ends = detect_phrases(segments)
        hammer = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(hammer) >= 1


class TestDJI0298_Small:
    """small model: merged everything into one huge segment.
    'Stand by' is buried inside but should still match."""

    def test_finds_standby(self) -> None:
        segments = _load("real_dji_0298_small")
        starts, ends = detect_phrases(segments)
        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) >= 1


class TestDJI0298_Large:
    """large-v3 model: better segmentation, 'Stand by' and 'Ready' visible."""

    def test_finds_start(self) -> None:
        segments = _load("real_dji_0298_large")
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1
