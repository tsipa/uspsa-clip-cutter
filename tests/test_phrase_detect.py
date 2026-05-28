"""Tests for phrase_detect module — pattern-based keyword matching."""

from video_stage_cutter.phrase_detect import PhraseMatch, detect_phrases
from video_stage_cutter.transcribe import TranscriptSegment, WordInfo


def _seg(start: float, end: float, text: str, words: list[tuple[float, float, str]] | None = None) -> TranscriptSegment:
    word_infos = []
    if words:
        word_infos = [WordInfo(start=s, end=e, word=w, probability=0.9) for s, e, w in words]
    return TranscriptSegment(start=start, end=end, text=text, words=word_infos)


class TestDetectPhrases:
    def test_finds_stand_by_and_end(self) -> None:
        segments = [
            _seg(10.0, 12.0, "are you ready", words=[(10.0, 10.5, "are"), (10.5, 11.0, "you"), (11.0, 12.0, "ready")]),
            _seg(12.0, 13.0, "stand by", words=[(12.0, 12.5, "stand"), (12.5, 13.0, "by")]),
            _seg(30.0, 35.0, "if clear hammer down and holster", words=[
                (30.0, 30.5, "if"), (30.5, 31.0, "clear"),
                (31.0, 31.5, "hammer"), (31.5, 32.0, "down"),
                (32.0, 32.5, "and"), (32.5, 33.0, "holster"),
            ]),
        ]
        starts, ends = detect_phrases(segments)
        assert len(starts) >= 1
        assert len(ends) >= 1

    def test_no_match_returns_empty(self) -> None:
        segments = [
            _seg(0.0, 5.0, "hello world how are things going today", words=[
                (0.0, 0.5, "hello"), (0.5, 1.0, "world"), (1.0, 1.5, "how"),
                (1.5, 2.0, "are"), (2.0, 2.5, "things"), (2.5, 3.0, "going"),
                (3.0, 3.5, "today"),
            ]),
        ]
        starts, ends = detect_phrases(segments)
        assert starts == []
        assert ends == []

    def test_fuzzy_match_misspelling(self) -> None:
        segments = [
            _seg(5.0, 7.0, "stand bye", words=[(5.0, 6.0, "stand"), (6.0, 7.0, "bye")]),
        ]
        starts, ends = detect_phrases(segments, threshold=65)
        assert len(starts) >= 1

    def test_fallback_segment_match(self) -> None:
        segments = [
            _seg(10.0, 13.0, "stand by", words=[]),
        ]
        starts, _ends = detect_phrases(segments)
        assert len(starts) >= 1

    def test_hammer_down_and_holster_detected(self) -> None:
        """Both 'hammer down' and 'holster' should be found as keywords
        and grouped into one end pattern."""
        segments = [
            _seg(30.0, 35.0, "hammer down and holster", words=[
                (30.0, 30.5, "hammer"), (30.5, 31.0, "down"),
                (31.0, 31.3, "and"), (31.3, 32.0, "holster"),
            ]),
        ]
        starts, ends = detect_phrases(segments)
        assert len(ends) == 1
        # pattern should contain both hammer down and holster keywords
        assert ends[0].score > 50  # high confidence from multiple keywords

    def test_holster_alone_low_confidence(self) -> None:
        """Standalone 'holster' should have low confidence (single keyword, weight=0.3)."""
        segments = [
            _seg(50.0, 51.0, "holster", words=[(50.0, 51.0, "holster")]),
        ]
        starts, ends = detect_phrases(segments)
        if ends:
            assert ends[0].score <= 35  # low confidence, just one keyword

    def test_no_cross_gap_match(self) -> None:
        """Words 30s apart should not form a keyword match."""
        segments = [
            _seg(10.0, 50.0, "by hammer down", words=[
                (10.0, 10.5, "by"),
                (40.0, 40.5, "hammer"),
                (40.5, 41.0, "down"),
            ]),
        ]
        starts, ends = detect_phrases(segments)
        for m in ends:
            assert m.start >= 40.0, f"Cross-gap match at {m.start}"

    def test_pattern_groups_nearby_keywords(self) -> None:
        """'are you ready' + 'stand by' close together = one high-confidence start."""
        segments = [
            _seg(8.0, 9.0, "are you ready", words=[
                (8.0, 8.3, "are"), (8.3, 8.6, "you"), (8.6, 9.0, "ready"),
            ]),
            _seg(10.0, 11.0, "stand by", words=[
                (10.0, 10.5, "stand"), (10.5, 11.0, "by"),
            ]),
        ]
        starts, ends = detect_phrases(segments)
        assert len(starts) == 1  # grouped into one pattern
        assert starts[0].score > 80  # high confidence

    def test_full_end_sequence_high_confidence(self) -> None:
        """Full end sequence with multiple keywords = very high confidence."""
        segments = [
            _seg(40.0, 48.0, "if finished unload show clear hammer down and holster", words=[
                (40.0, 40.5, "if"), (40.5, 41.0, "finished"),
                (41.0, 41.5, "unload"), (41.5, 42.0, "show"), (42.0, 42.5, "clear"),
                (42.5, 43.0, "hammer"), (43.0, 43.5, "down"),
                (43.5, 43.8, "and"), (43.8, 44.5, "holster"),
            ]),
        ]
        starts, ends = detect_phrases(segments)
        assert len(ends) >= 1
        assert ends[0].score > 90  # very high confidence
