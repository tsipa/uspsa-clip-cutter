"""Tests using synthetic JSON transcript fixtures.

No faster-whisper or ffmpeg required. Tests phrase detection,
stage assembly, and fallback/overlap logic against pre-built transcripts.
"""

from __future__ import annotations

import json
from pathlib import Path

from video_stage_cutter.phrase_detect import detect_phrases
from video_stage_cutter.pipeline import (
    Anchor,
    _assemble_stages,
    _subtract_intervals,
    _trim_fallback,
    Stage,
)
from video_stage_cutter.transcribe import TranscriptSegment, WordInfo

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcripts"


def load_transcript_fixture(name: str) -> list[TranscriptSegment]:
    """Load a fixture JSON and return a list of TranscriptSegments."""
    path = FIXTURES_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: list[TranscriptSegment] = []
    for seg in data["segments"]:
        words = [
            WordInfo(
                start=w["start"],
                end=w["end"],
                word=w["word"],
                probability=w["probability"],
            )
            for w in seg.get("words", [])
        ]
        segments.append(TranscriptSegment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            words=words,
        ))
    return segments


# ---------------------------------------------------------------------------
# Phrase detection on fixtures
# ---------------------------------------------------------------------------

class TestNormalTranscript:
    def test_finds_standby_and_end(self) -> None:
        segments = load_transcript_fixture("stage_normal")
        starts, ends = detect_phrases(segments)

        assert len(starts) >= 1
        assert len(ends) >= 1

        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) >= 1
        # sliding window may start from neighboring word, so allow from 8.0
        assert 8.0 <= standby[0].start <= 10.0

        end_cmds = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(end_cmds) >= 1
        assert end_cmds[0].end >= 38.0, f"End command end={end_cmds[0].end}, expected >=38.0"


class TestNoEndTranscript:
    def test_finds_start_but_no_end(self) -> None:
        segments = load_transcript_fixture("stage_no_end")
        starts, ends = detect_phrases(segments)

        assert len(starts) >= 1
        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) >= 1

        hammer_ends = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(hammer_ends) == 0


class TestNoStartTranscript:
    def test_finds_end_but_no_start(self) -> None:
        segments = load_transcript_fixture("stage_no_start")
        starts, ends = detect_phrases(segments)

        assert len(starts) == 0

        assert len(ends) >= 1
        assert any("hammer" in m.matched_phrase.lower() for m in ends)


class TestFuzzyBadAsr:
    def test_still_matches_despite_asr_errors(self) -> None:
        segments = load_transcript_fixture("stage_fuzzy_bad_asr")
        starts, ends = detect_phrases(segments, threshold=60)

        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) >= 1, "Should match 'stand bye' as 'stand by'"

        hammer_ends = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(hammer_ends) >= 1, "Should match 'hammer dawn' as 'hammer down'"


class TestMultipleStages:
    def test_finds_multiple_starts_and_ends(self) -> None:
        segments = load_transcript_fixture("stage_multiple_stages")
        starts, ends = detect_phrases(segments)

        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        assert len(standby) >= 2, f"Expected >=2 standby, got {len(standby)}"

        hammer_ends = [m for m in ends if "hammer" in m.matched_phrase.lower()]
        assert len(hammer_ends) >= 2, f"Expected >=2 end commands, got {len(hammer_ends)}"

        assert standby[0].start < standby[1].start
        assert hammer_ends[0].start < hammer_ends[1].start


class TestFalsePositiveNoise:
    def test_does_not_create_confirmed_stage(self) -> None:
        """Sentences that contain 'are you', 'stand', 'hammer down' in
        non-RO context should not produce high-confidence matches that
        would form a confirmed stage."""
        segments = load_transcript_fixture("stage_false_positive_noise")
        starts, ends = detect_phrases(segments)

        # "Are you going to paste" should NOT match "are you ready" well
        ready_matches = [m for m in starts if "ready" in m.matched_phrase.lower()]
        high_conf_ready = [m for m in ready_matches if m.score >= 80]
        assert len(high_conf_ready) == 0, (
            f"False positive: 'Are you going to paste' matched 'are you ready' "
            f"with score {high_conf_ready[0].score if high_conf_ready else 0}"
        )

        # Even if something matched, there should be no standby+end pairing
        standby = [m for m in starts if "stand by" in m.matched_phrase.lower()]
        high_conf_standby = [m for m in standby if m.score >= 80]
        assert len(high_conf_standby) == 0, "Stand over there should not match stand by at >=80"


# ---------------------------------------------------------------------------
# Stage assembly with fixture-derived anchors
# ---------------------------------------------------------------------------

class TestAssemblyFromFixtures:
    def test_normal_produces_one_confirmed_stage(self) -> None:
        segments = load_transcript_fixture("stage_normal")
        starts, ends = detect_phrases(segments)

        anchors = _build_anchors(starts, ends, file_idx=0, epoch=0.0)

        # add synthetic beep after standby (assembly requires a beep anchor)
        standby_a = [a for a in anchors if a.kind == "standby"]
        if standby_a:
            beep_time = standby_a[-1].end_offset + 1.0
            anchors.append(Anchor(
                kind="beep", abs_time=beep_time, file_idx=0,
                file_offset=beep_time, text="timer_beep", score=80,
            ))

        stages = _assemble_stages(anchors, min_clip_length=5.0)

        confirmed = [s for s in stages if s.complete]
        assert len(confirmed) >= 1

    def test_no_end_produces_fallback(self) -> None:
        segments = load_transcript_fixture("stage_no_end")
        starts, ends = detect_phrases(segments)

        anchors = _build_anchors(starts, ends, file_idx=0, epoch=0.0)
        # simulate a beep 1s after standby
        standby_a = [a for a in anchors if a.kind == "standby"]
        if standby_a:
            beep_time = standby_a[0].end_offset + 1.0
            anchors.append(Anchor(
                kind="beep", abs_time=beep_time, file_idx=0,
                file_offset=beep_time, text="timer_beep", score=80,
            ))

        stages = _assemble_stages(anchors, min_clip_length=5.0)
        assert len(stages) >= 1
        assert any(not s.complete for s in stages)
        fallbacks = [s for s in stages if not s.complete]
        assert "fallback" in fallbacks[0].end_reason

    def test_no_start_produces_fallback(self) -> None:
        segments = load_transcript_fixture("stage_no_start")
        starts, ends = detect_phrases(segments)

        anchors = _build_anchors(starts, ends, file_idx=0, epoch=0.0)
        stages = _assemble_stages(anchors, min_clip_length=5.0)

        assert len(stages) >= 1
        fallbacks = [s for s in stages if not s.complete]
        assert len(fallbacks) >= 1
        assert "no_start" in fallbacks[0].start_reason

    def test_multiple_stages_produces_two_confirmed(self) -> None:
        segments = load_transcript_fixture("stage_multiple_stages")
        starts, ends = detect_phrases(segments)

        anchors = _build_anchors(starts, ends, file_idx=0, epoch=0.0)
        # add beeps after each standby
        standby_a = [a for a in anchors if a.kind == "standby"]
        for sb in standby_a:
            beep_time = sb.end_offset + 1.0
            anchors.append(Anchor(
                kind="beep", abs_time=beep_time, file_idx=0,
                file_offset=beep_time, text="timer_beep", score=80,
            ))

        stages = _assemble_stages(anchors, min_clip_length=5.0)
        confirmed = [s for s in stages if s.complete]
        assert len(confirmed) >= 2


def _build_anchors(
    starts: list,
    ends: list,
    file_idx: int,
    epoch: float,
) -> list[Anchor]:
    """Convert PhraseMatch lists into Anchor lists for assembly testing."""
    anchors: list[Anchor] = []
    for m in starts:
        kind = "standby" if "stand by" in m.matched_phrase.lower() else "ready"
        anchors.append(Anchor(
            kind=kind, abs_time=epoch + m.start, file_idx=file_idx,
            file_offset=m.start, text=m.text, score=m.score,
            end_offset=m.end,
        ))
    for m in ends:
        anchors.append(Anchor(
            kind="end_command", abs_time=epoch + m.start, file_idx=file_idx,
            file_offset=m.start, text=m.text, score=m.score,
            end_offset=m.end,
        ))
    return anchors


# ---------------------------------------------------------------------------
# Fallback overlap interval tests (using fixtures context)
# ---------------------------------------------------------------------------

class TestFallbackOverlapWithFixtureContext:
    def test_confirmed_10_45_blocks_fallback_0_55(self) -> None:
        """Confirmed 10-45, fallback 0-55 => fallback trimmed to 45-55."""
        confirmed_intervals = [(10.0, 45.0)]
        remaining = _subtract_intervals(0.0, 55.0, confirmed_intervals)
        assert (0.0, 10.0) in remaining
        assert (45.0, 55.0) in remaining
        # longest is 0-10 (10s)
        longest = max(remaining, key=lambda iv: iv[1] - iv[0])
        assert longest == (0.0, 10.0) or longest == (45.0, 55.0)

    def test_confirmed_130_170_blocks_fallback_60_240(self) -> None:
        """Confirmed 130-170, fallback 60-240 => remaining 60-130 and 170-240."""
        confirmed_intervals = [(130.0, 170.0)]
        remaining = _subtract_intervals(60.0, 240.0, confirmed_intervals)
        assert (60.0, 130.0) in remaining
        assert (170.0, 240.0) in remaining
        longest = max(remaining, key=lambda iv: iv[1] - iv[0])
        assert longest[1] - longest[0] == 70.0

    def test_fallback_skipped_when_remaining_too_short(self) -> None:
        stage = Stage(
            beep=Anchor(kind="beep", abs_time=10.0, file_idx=0, file_offset=10.0,
                        text="timer_beep", score=80),
            clip_start=10.0,
            clip_end=190.0,
            start_reason="beep",
            end_reason="fallback_3min_no_end",
            complete=False,
        )
        # confirmed covers almost all of it
        confirmed_intervals = [(12.0, 188.0)]
        result = _trim_fallback(stage, confirmed_intervals, min_clip_length=5.0)
        assert result is None
