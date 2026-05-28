"""Fuzzy phrase matching over transcript segments."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rapidfuzz import fuzz

from video_stage_cutter.transcribe import TranscriptSegment

log = logging.getLogger(__name__)

START_PHRASES = [
    "are you ready",
    "stand by",
    "standby",
    "make ready",
    "load and make ready",
]

END_PHRASES = [
    "if you are finished unload and show clear",
    "unload and show clear",
    "if clear hammer down and holster",
    "if clear hammer down",
    "hammer down and holster",
    "hammer down",
    "holster",
    "range is clear",
    "stage is clear",
]

DEFAULT_THRESHOLD = 70


@dataclass
class PhraseMatch:
    start: float
    end: float
    text: str
    score: float
    matched_phrase: str
    role: str  # "start" or "end"


def _sliding_window_match(
    words_text: str,
    phrase: str,
    words_starts: list[float],
    words_ends: list[float],
    word_strings: list[str],
    threshold: float,
    role: str,
) -> list[PhraseMatch]:
    """Slide a window over the word sequence and score against *phrase*."""
    matches: list[PhraseMatch] = []
    phrase_word_count = len(phrase.split())

    for window_size in range(max(1, phrase_word_count - 1), phrase_word_count + 3):
        if window_size > len(word_strings):
            continue
        for i in range(len(word_strings) - window_size + 1):
            window_text = " ".join(word_strings[i : i + window_size])
            score = fuzz.ratio(window_text.lower(), phrase.lower())
            if score >= threshold:
                matches.append(PhraseMatch(
                    start=words_starts[i],
                    end=words_ends[i + window_size - 1],
                    text=window_text,
                    score=score,
                    matched_phrase=phrase,
                    role=role,
                ))
    return matches


def detect_phrases(
    segments: list[TranscriptSegment],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[PhraseMatch], list[PhraseMatch]]:
    """Find start-phrase and end-phrase candidates in *segments*.

    Returns ``(start_matches, end_matches)`` sorted by timestamp.
    """
    all_words: list[str] = []
    all_starts: list[float] = []
    all_ends: list[float] = []

    for seg in segments:
        for w in seg.words:
            cleaned = w.word.strip().strip(".,!?;:'\"").lower()
            if cleaned:
                all_words.append(cleaned)
                all_starts.append(w.start)
                all_ends.append(w.end)

    if not all_words:
        segment_text = " ".join(s.text for s in segments).lower()
        return _fallback_segment_match(segments, segment_text, threshold)

    start_matches: list[PhraseMatch] = []
    end_matches: list[PhraseMatch] = []

    for phrase in START_PHRASES:
        start_matches.extend(
            _sliding_window_match(
                "", phrase, all_starts, all_ends, all_words, threshold, "start",
            )
        )

    for phrase in END_PHRASES:
        end_matches.extend(
            _sliding_window_match(
                "", phrase, all_starts, all_ends, all_words, threshold, "end",
            )
        )

    start_matches = _deduplicate(start_matches)
    end_matches = _deduplicate(end_matches)

    start_matches.sort(key=lambda m: m.start)
    end_matches.sort(key=lambda m: m.start)

    for m in start_matches:
        log.info(
            "  START candidate: '%.50s' matched '%s' score=%.0f at %.2f–%.2fs",
            m.text, m.matched_phrase, m.score, m.start, m.end,
        )
    for m in end_matches:
        log.info(
            "  END   candidate: '%.50s' matched '%s' score=%.0f at %.2f–%.2fs",
            m.text, m.matched_phrase, m.score, m.start, m.end,
        )
    if not start_matches:
        log.warning("  No start phrases found in transcript")
    if not end_matches:
        log.warning("  No end phrases found in transcript")

    return start_matches, end_matches


def _fallback_segment_match(
    segments: list[TranscriptSegment],
    _full_text: str,
    threshold: float,
) -> tuple[list[PhraseMatch], list[PhraseMatch]]:
    """Match against whole segment text when word-level timestamps are unavailable."""
    start_matches: list[PhraseMatch] = []
    end_matches: list[PhraseMatch] = []

    for seg in segments:
        seg_lower = seg.text.lower()
        for phrase in START_PHRASES:
            score = fuzz.partial_ratio(seg_lower, phrase)
            if score >= threshold:
                start_matches.append(PhraseMatch(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    score=score,
                    matched_phrase=phrase,
                    role="start",
                ))
        for phrase in END_PHRASES:
            score = fuzz.partial_ratio(seg_lower, phrase)
            if score >= threshold:
                end_matches.append(PhraseMatch(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    score=score,
                    matched_phrase=phrase,
                    role="end",
                ))

    return start_matches, end_matches


def _deduplicate(matches: list[PhraseMatch], time_tolerance: float = 3.0) -> list[PhraseMatch]:
    """Keep only the highest-scoring match within each *time_tolerance* window.

    Only deduplicates matches of the same base phrase (e.g. two variants of
    'hammer down' at the same time), not different phrases that happen to be close.
    """
    if not matches:
        return matches
    matches.sort(key=lambda m: m.start)
    result: list[PhraseMatch] = [matches[0]]
    for m in matches[1:]:
        # only dedup if overlapping in time AND sharing the shorter phrase
        same_area = m.start - result[-1].start < time_tolerance
        phrases_overlap = (
            m.matched_phrase.lower() in result[-1].matched_phrase.lower()
            or result[-1].matched_phrase.lower() in m.matched_phrase.lower()
        )
        if same_area and phrases_overlap:
            if m.score > result[-1].score:
                result[-1] = m
        else:
            result.append(m)
    return result
