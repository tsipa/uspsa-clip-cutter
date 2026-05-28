"""Pattern-based detection of USPSA stage boundaries.

Instead of matching full phrases, detects individual keywords and groups
them into start/end sequences. Each keyword in the pattern boosts confidence.

Start sequence (in order, ~60s window):
  "make ready" → "are you ready" → "stand by" → beep

End sequence (in order, ~30s window):
  "finished" → "unload" → "clear" → "hammer down" → "holster" → "range is clear"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from video_stage_cutter.transcribe import TranscriptSegment

log = logging.getLogger(__name__)

# Keywords to search for. Each is (keyword, fuzzy_threshold, weight).
# Weight determines how much this keyword contributes to pattern confidence.

START_KEYWORDS: list[tuple[str, int, float]] = [
    ("make ready", 80, 0.3),
    ("are you ready", 75, 0.5),
    ("ready", 80, 0.2),
    ("stand by", 80, 0.8),
    ("standby", 80, 0.8),
]

END_KEYWORDS: list[tuple[str, int, float]] = [
    ("finished", 80, 0.2),
    ("unload", 80, 0.3),
    ("show clear", 75, 0.3),
    ("hammer down", 75, 0.6),
    ("hammer", 80, 0.3),
    ("holster", 80, 0.3),
    ("range is clear", 75, 0.5),
    ("stage is clear", 75, 0.5),
]

START_WINDOW = 60.0
END_WINDOW = 30.0
MAX_WORD_GAP = 2.0
DEFAULT_THRESHOLD = 70


@dataclass
class KeywordHit:
    keyword: str
    start: float
    end: float
    text: str
    score: float
    weight: float
    word_indices: tuple[int, ...]


@dataclass
class PatternMatch:
    """A detected start or end pattern with constituent keyword hits."""
    start: float
    end: float
    hits: list[KeywordHit]
    confidence: float
    role: str  # "start" or "end"
    text: str = ""

    @property
    def best_keyword(self) -> str:
        if not self.hits:
            return ""
        return max(self.hits, key=lambda h: h.weight).keyword


@dataclass
class PhraseMatch:
    """Backward-compatible match result used by pipeline."""
    start: float
    end: float
    text: str
    score: float
    matched_phrase: str
    role: str


def detect_phrases(
    segments: list[TranscriptSegment],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[PhraseMatch], list[PhraseMatch]]:
    """Find start and end pattern candidates in transcript.

    Returns (start_matches, end_matches) sorted by timestamp.
    Each match represents a keyword cluster pattern.
    """
    all_words, all_starts, all_ends = _extract_words(segments)

    if not all_words:
        return _fallback_segment_match(segments, threshold)

    start_hits = _find_keyword_hits(all_words, all_starts, all_ends, START_KEYWORDS, threshold)
    end_hits = _find_keyword_hits(all_words, all_starts, all_ends, END_KEYWORDS, threshold)

    log.info("  Keyword hits: %d start, %d end", len(start_hits), len(end_hits))
    for h in start_hits:
        log.info("    START kw='%s' text='%s' score=%.0f weight=%.1f at %.2f-%.2fs",
                 h.keyword, h.text, h.score, h.weight, h.start, h.end)
    for h in end_hits:
        log.info("    END   kw='%s' text='%s' score=%.0f weight=%.1f at %.2f-%.2fs",
                 h.keyword, h.text, h.score, h.weight, h.start, h.end)

    start_patterns = _group_into_patterns(start_hits, START_WINDOW, "start")
    end_patterns = _group_into_patterns(end_hits, END_WINDOW, "end")

    start_matches = [_pattern_to_phrase_match(p) for p in start_patterns]
    end_matches = [_pattern_to_phrase_match(p) for p in end_patterns]

    start_matches.sort(key=lambda m: m.start)
    end_matches.sort(key=lambda m: m.start)

    for m in start_matches:
        log.info("  START pattern: '%.60s' confidence=%.2f at %.2f-%.2fs",
                 m.text, m.score, m.start, m.end)
    for m in end_matches:
        log.info("  END   pattern: '%.60s' confidence=%.2f at %.2f-%.2fs",
                 m.text, m.score, m.start, m.end)

    if not start_matches:
        log.warning("  No start patterns found in transcript")
    if not end_matches:
        log.warning("  No end patterns found in transcript")

    return start_matches, end_matches


def _extract_words(
    segments: list[TranscriptSegment],
) -> tuple[list[str], list[float], list[float]]:
    words: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    for seg in segments:
        for w in seg.words:
            cleaned = w.word.strip().strip(".,!?;:'\"").lower()
            if cleaned:
                words.append(cleaned)
                starts.append(w.start)
                ends.append(w.end)
    return words, starts, ends


def _find_keyword_hits(
    words: list[str],
    word_starts: list[float],
    word_ends: list[float],
    keywords: list[tuple[str, int, float]],
    base_threshold: float,
) -> list[KeywordHit]:
    """Find all keyword occurrences using sliding window."""
    hits: list[KeywordHit] = []

    for keyword, kw_threshold, weight in keywords:
        kw_word_count = len(keyword.split())
        effective_threshold = max(base_threshold, kw_threshold)

        for window_size in range(kw_word_count, kw_word_count + 2):
            if window_size > len(words):
                continue
            for i in range(len(words) - window_size + 1):
                # check time gap
                has_gap = False
                for k in range(i, i + window_size - 1):
                    if word_starts[k + 1] - word_ends[k] > MAX_WORD_GAP:
                        has_gap = True
                        break
                if has_gap:
                    continue

                window_text = " ".join(words[i : i + window_size])
                score = fuzz.ratio(window_text, keyword)
                if score >= effective_threshold:
                    hits.append(KeywordHit(
                        keyword=keyword,
                        start=word_starts[i],
                        end=word_ends[i + window_size - 1],
                        text=window_text,
                        score=score,
                        weight=weight,
                        word_indices=tuple(range(i, i + window_size)),
                    ))

    # dedup: if multiple keywords match the same word indices, keep highest weight
    hits = _dedup_keyword_hits(hits)
    return hits


def _dedup_keyword_hits(hits: list[KeywordHit]) -> list[KeywordHit]:
    """Remove overlapping keyword hits, keeping highest weight."""
    if not hits:
        return hits
    hits.sort(key=lambda h: (h.weight, h.score), reverse=True)
    used: set[int] = set()
    result: list[KeywordHit] = []
    for h in hits:
        indices = set(h.word_indices)
        if indices & used:
            continue
        used |= indices
        result.append(h)
    result.sort(key=lambda h: h.start)
    return result


def _group_into_patterns(
    hits: list[KeywordHit],
    window: float,
    role: str,
) -> list[PatternMatch]:
    """Group nearby keyword hits into patterns within *window* seconds."""
    if not hits:
        return []

    hits.sort(key=lambda h: h.start)
    patterns: list[PatternMatch] = []

    i = 0
    while i < len(hits):
        cluster = [hits[i]]
        j = i + 1
        while j < len(hits) and hits[j].start - cluster[0].start <= window:
            cluster.append(hits[j])
            j += 1

        # deduplicate keywords in cluster (keep best score per keyword)
        best_per_kw: dict[str, KeywordHit] = {}
        for h in cluster:
            if h.keyword not in best_per_kw or h.score > best_per_kw[h.keyword].score:
                best_per_kw[h.keyword] = h
        unique_hits = sorted(best_per_kw.values(), key=lambda h: h.start)

        confidence = sum(h.weight * (h.score / 100.0) for h in unique_hits)
        confidence = min(1.0, confidence)

        pattern_start = min(h.start for h in unique_hits)
        pattern_end = max(h.end for h in unique_hits)
        text = " + ".join(f"{h.keyword}({h.score:.0f})" for h in unique_hits)

        patterns.append(PatternMatch(
            start=pattern_start,
            end=pattern_end,
            hits=unique_hits,
            confidence=confidence,
            role=role,
            text=text,
        ))

        i = j

    return patterns


def _pattern_to_phrase_match(pattern: PatternMatch) -> PhraseMatch:
    """Convert a PatternMatch to backward-compatible PhraseMatch."""
    return PhraseMatch(
        start=pattern.start,
        end=pattern.end,
        text=pattern.text,
        score=pattern.confidence * 100,
        matched_phrase=pattern.best_keyword,
        role=pattern.role,
    )


def _fallback_segment_match(
    segments: list[TranscriptSegment],
    threshold: float,
) -> tuple[list[PhraseMatch], list[PhraseMatch]]:
    """Match against whole segment text when word-level timestamps are unavailable."""
    start_matches: list[PhraseMatch] = []
    end_matches: list[PhraseMatch] = []

    start_phrases = [kw for kw, _, _ in START_KEYWORDS]
    end_phrases = [kw for kw, _, _ in END_KEYWORDS]

    for seg in segments:
        seg_lower = seg.text.lower()
        for phrase in start_phrases:
            score = fuzz.partial_ratio(seg_lower, phrase)
            if score >= threshold:
                start_matches.append(PhraseMatch(
                    start=seg.start, end=seg.end, text=seg.text,
                    score=score, matched_phrase=phrase, role="start",
                ))
        for phrase in end_phrases:
            score = fuzz.partial_ratio(seg_lower, phrase)
            if score >= threshold:
                end_matches.append(PhraseMatch(
                    start=seg.start, end=seg.end, text=seg.text,
                    score=score, matched_phrase=phrase, role="end",
                ))

    return start_matches, end_matches
