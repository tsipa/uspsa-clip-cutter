"""Tests for phrase_detect module."""

from video_stage_cutter.phrase_detect import PhraseMatch, _deduplicate, detect_phrases
from video_stage_cutter.transcribe import TranscriptSegment, WordInfo


def _seg(start: float, end: float, text: str, words: list[tuple[float, float, str]] | None = None) -> TranscriptSegment:
    word_infos = []
    if words:
        word_infos = [WordInfo(start=s, end=e, word=w, probability=0.9) for s, e, w in words]
    return TranscriptSegment(start=start, end=end, text=text, words=word_infos)


class TestDetectPhrases:
    def test_finds_stand_by(self) -> None:
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
        assert any("stand by" in m.matched_phrase for m in starts)
        assert any("hammer down" in m.matched_phrase for m in ends)

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
        assert starts[0].score >= 65

    def test_fallback_segment_match(self) -> None:
        segments = [
            _seg(10.0, 13.0, "stand by", words=[]),
        ]
        starts, _ends = detect_phrases(segments)
        assert len(starts) >= 1


class TestDeduplicate:
    def test_keeps_best_in_window(self) -> None:
        matches = [
            PhraseMatch(start=10.0, end=11.0, text="stand by", score=80, matched_phrase="stand by", role="start"),
            PhraseMatch(start=10.1, end=11.1, text="stand bye", score=90, matched_phrase="stand by", role="start"),
        ]
        result = _deduplicate(matches, time_tolerance=0.5)
        assert len(result) == 1
        assert result[0].score == 90

    def test_keeps_separate_windows(self) -> None:
        matches = [
            PhraseMatch(start=10.0, end=11.0, text="stand by", score=80, matched_phrase="stand by", role="start"),
            PhraseMatch(start=20.0, end=21.0, text="stand by", score=85, matched_phrase="stand by", role="start"),
        ]
        result = _deduplicate(matches, time_tolerance=0.5)
        assert len(result) == 2
