"""Sequence-based detection of USPSA stage boundaries.

RO commands follow a strict order. We detect individual keywords then
find the longest ordered subsequence matching the expected protocol.

Start sequence:
  make ready → are you ready → stand by → [beep]

End sequence:
  finished → unload → show clear → hammer down → holster → range is clear

Keywords in correct order boost confidence. Out-of-order keywords are
ignored. Missing steps reduce confidence but don't block detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from video_stage_cutter.transcribe import TranscriptSegment

log = logging.getLogger(__name__)

MAX_WORD_GAP = 2.0
DEFAULT_THRESHOLD = 70


@dataclass
class SequenceStep:
    """One step in the expected RO command sequence."""
    order: int
    keywords: list[str]
    threshold: int
    weight: float


# Steps must be in chronological order. Order values define the expected sequence.
START_SEQUENCE: list[SequenceStep] = [
    SequenceStep(order=0, keywords=["load and make ready", "make ready"], threshold=80, weight=0.15),
    SequenceStep(order=1, keywords=["are you ready"], threshold=75, weight=0.25),
    SequenceStep(order=2, keywords=["ready"], threshold=85, weight=0.10),
    SequenceStep(order=3, keywords=["stand by", "standby"], threshold=75, weight=0.40),
]

END_SEQUENCE: list[SequenceStep] = [
    SequenceStep(order=0, keywords=["finished"], threshold=80, weight=0.10),
    SequenceStep(order=1, keywords=["unload"], threshold=80, weight=0.15),
    SequenceStep(order=2, keywords=["show clear"], threshold=75, weight=0.15),
    SequenceStep(order=3, keywords=["hammer down"], threshold=75, weight=0.25),
    SequenceStep(order=4, keywords=["hammer"], threshold=85, weight=0.10),
    SequenceStep(order=5, keywords=["holster"], threshold=85, weight=0.10),
    SequenceStep(order=6, keywords=["range is clear", "stage is clear"], threshold=75, weight=0.15),
]

START_WINDOW = 90.0
END_WINDOW = 30.0


@dataclass
class KeywordHit:
    keyword: str
    step_order: int
    start: float
    end: float
    text: str
    score: float
    weight: float
    word_indices: tuple[int, ...]


@dataclass
class SequenceMatch:
    """A detected ordered sequence of RO keywords."""
    start: float
    end: float
    hits: list[KeywordHit]
    confidence: float
    role: str
    text: str = ""
    steps_found: int = 0
    steps_total: int = 0
    in_order: bool = True


@dataclass
class PhraseMatch:
    """Backward-compatible match result used by pipeline."""
    start: float
    end: float
    text: str
    score: float
    matched_phrase: str
    role: str
    steps_found: int = 0
    steps_total: int = 0
    matched_step_orders: tuple[int, ...] = ()


def detect_phrases(
    segments: list[TranscriptSegment],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[PhraseMatch], list[PhraseMatch]]:
    """Detect start/end sequences in transcript."""
    all_words, all_starts, all_ends = _extract_words(segments)

    if not all_words:
        return _fallback_segment_match(segments, threshold)

    start_hits = _find_all_hits(all_words, all_starts, all_ends, START_SEQUENCE, threshold)
    end_hits = _find_all_hits(all_words, all_starts, all_ends, END_SEQUENCE, threshold)

    log.info("  Keyword hits: %d start, %d end", len(start_hits), len(end_hits))
    for h in start_hits:
        log.info("    START step=%d kw='%s' text='%s' score=%.0f at %.2f-%.2fs",
                 h.step_order, h.keyword, h.text, h.score, h.start, h.end)
    for h in end_hits:
        log.info("    END   step=%d kw='%s' text='%s' score=%.0f at %.2f-%.2fs",
                 h.step_order, h.keyword, h.text, h.score, h.start, h.end)

    start_seqs = _find_sequences(start_hits, START_SEQUENCE, START_WINDOW, "start")
    end_seqs = _find_sequences(end_hits, END_SEQUENCE, END_WINDOW, "end")

    start_matches = [_seq_to_phrase_match(s) for s in start_seqs]
    end_matches = [_seq_to_phrase_match(s) for s in end_seqs]

    start_matches.sort(key=lambda m: m.start)
    end_matches.sort(key=lambda m: m.start)

    for m in start_matches:
        log.info("  START seq: '%.80s' confidence=%.2f at %.2f-%.2fs",
                 m.text, m.score / 100, m.start, m.end)
    for m in end_matches:
        log.info("  END   seq: '%.80s' confidence=%.2f at %.2f-%.2fs",
                 m.text, m.score / 100, m.start, m.end)

    if not start_matches:
        log.warning("  No start sequences found")
    if not end_matches:
        log.warning("  No end sequences found")

    return start_matches, end_matches


def _extract_words(
    segments: list[TranscriptSegment],
) -> tuple[list[str], list[float], list[float]]:
    words, starts, ends = [], [], []
    for seg in segments:
        for w in seg.words:
            cleaned = w.word.strip().strip(".,!?;:'\"").lower()
            if cleaned:
                words.append(cleaned)
                starts.append(w.start)
                ends.append(w.end)
    return words, starts, ends


_MIN_FRAGMENT_WORD_LEN = 4


def _keyword_fragments(keyword: str) -> list[tuple[str, float]]:
    """Generate sub-phrase fragments from a multi-word keyword.

    Each fragment carries a weight multiplier (0.0-1.0) reflecting how
    much of the original phrase it covers. Shorter fragments = weaker
    evidence but still useful when Whisper garbles part of the phrase.

    "range is clear" →
        ("range is clear", 1.0),   # full phrase
        ("is clear", 0.50),        # 2-word suffix
        ("range is", 0.50),        # 2-word prefix
        ("range", 0.20),           # single word (≥4 chars only)
        ("clear", 0.20),           # single word
    """
    kw_words = keyword.split()
    n = len(kw_words)
    if n <= 1:
        return [(keyword, 1.0)]

    fragments: list[tuple[str, float]] = [(keyword, 1.0)]

    for length in range(n - 1, 1, -1):
        multiplier = (length / n) * 0.75
        seen: set[str] = set()
        for start in range(n - length + 1):
            frag = " ".join(kw_words[start : start + length])
            if frag not in seen:
                seen.add(frag)
                fragments.append((frag, multiplier))

    for word in kw_words:
        if len(word) >= _MIN_FRAGMENT_WORD_LEN:
            fragments.append((word, 0.20))

    return fragments


def _find_all_hits(
    words: list[str],
    word_starts: list[float],
    word_ends: list[float],
    sequence: list[SequenceStep],
    base_threshold: float,
) -> list[KeywordHit]:
    """Find all keyword occurrences for all steps, including partial fragments."""
    all_hits: list[KeywordHit] = []

    for step in sequence:
        for keyword in step.keywords:
            for fragment, weight_mult in _keyword_fragments(keyword):
                frag_len = len(fragment.split())
                frag_weight = step.weight * weight_mult

                if weight_mult < 0.3:
                    effective_threshold = max(base_threshold, step.threshold, 90)
                elif weight_mult < 1.0:
                    effective_threshold = max(base_threshold, step.threshold, 80)
                else:
                    effective_threshold = max(base_threshold, step.threshold)

                for window_size in range(frag_len, frag_len + 2):
                    if window_size > len(words):
                        continue
                    for i in range(len(words) - window_size + 1):
                        has_gap = False
                        for k in range(i, i + window_size - 1):
                            if word_starts[k + 1] - word_ends[k] > MAX_WORD_GAP:
                                has_gap = True
                                break
                        if has_gap:
                            continue

                        window_text = " ".join(words[i : i + window_size])
                        score = fuzz.ratio(window_text, fragment)
                        if score >= effective_threshold:
                            all_hits.append(KeywordHit(
                                keyword=keyword,
                                step_order=step.order,
                                start=word_starts[i],
                                end=word_ends[i + window_size - 1],
                                text=window_text,
                                score=score,
                                weight=frag_weight,
                                word_indices=tuple(range(i, i + window_size)),
                            ))

    all_hits = _dedup_hits(all_hits)
    return all_hits


def _dedup_hits(hits: list[KeywordHit]) -> list[KeywordHit]:
    """Remove overlapping hits, keeping highest weight then score."""
    if not hits:
        return hits
    hits.sort(key=lambda h: (h.weight, h.score), reverse=True)
    used: set[int] = set()
    result: list[KeywordHit] = []
    for h in hits:
        if set(h.word_indices) & used:
            continue
        used |= set(h.word_indices)
        result.append(h)
    result.sort(key=lambda h: h.start)
    return result


def _find_sequences(
    hits: list[KeywordHit],
    sequence: list[SequenceStep],
    window: float,
    role: str,
) -> list[SequenceMatch]:
    """Find ordered subsequences of keyword hits.

    For each potential anchor hit, try to build the longest ordered
    subsequence where each subsequent keyword appears AFTER the
    previous one in time and has a higher or equal step order.
    """
    if not hits:
        return []

    hits.sort(key=lambda h: h.start)
    total_steps = len(set(s.order for s in sequence))
    sequences: list[SequenceMatch] = []
    used_hits: set[int] = set()  # index into hits

    # try starting from each hit, build the longest ordered sequence
    for anchor_idx, anchor in enumerate(hits):
        if anchor_idx in used_hits:
            continue

        # collect hits within window, in order
        chain = [anchor]
        last_order = anchor.step_order
        last_time = anchor.end

        for j in range(anchor_idx + 1, len(hits)):
            if j in used_hits:
                continue
            h = hits[j]
            if h.start - anchor.start > window:
                break
            # must be same or later step in sequence AND later in time
            if h.step_order >= last_order and h.start >= last_time - 0.5:
                # skip if same step already in chain
                if h.step_order == last_order and any(c.step_order == h.step_order for c in chain):
                    continue
                chain.append(h)
                last_order = h.step_order
                last_time = h.end

        # calculate confidence from ordered chain
        steps_found = len(set(h.step_order for h in chain))
        confidence = sum(h.weight * (h.score / 100.0) for h in chain)
        confidence = min(1.0, confidence)

        # bonus for multiple steps in correct order
        if steps_found >= 2:
            order_bonus = 0.1 * (steps_found - 1)
            confidence = min(1.0, confidence + order_bonus)
            log.info("    Sequence order bonus: %d steps in order → +%.2f",
                     steps_found, order_bonus)

        seq_start = min(h.start for h in chain)
        seq_end = max(h.end for h in chain)
        text = " → ".join(f"{h.keyword}({h.score:.0f})@{h.start:.1f}s" for h in chain)

        seq = SequenceMatch(
            start=seq_start,
            end=seq_end,
            hits=chain,
            confidence=confidence,
            role=role,
            text=text,
            steps_found=steps_found,
            steps_total=total_steps,
            in_order=True,
        )
        sequences.append(seq)

        # mark used
        for j, h in enumerate(hits):
            if h in chain:
                used_hits.add(j)

    return sequences


def _seq_to_phrase_match(seq: SequenceMatch) -> PhraseMatch:
    best = max(seq.hits, key=lambda h: h.weight)
    return PhraseMatch(
        start=seq.start,
        end=seq.end,
        text=seq.text,
        score=seq.confidence * 100,
        matched_phrase=best.keyword,
        role=seq.role,
        steps_found=seq.steps_found,
        steps_total=seq.steps_total,
        matched_step_orders=tuple(sorted(set(h.step_order for h in seq.hits))),
    )


def _fallback_segment_match(
    segments: list[TranscriptSegment],
    threshold: float,
) -> tuple[list[PhraseMatch], list[PhraseMatch]]:
    """Match against whole segment text when word-level timestamps are unavailable."""
    start_matches: list[PhraseMatch] = []
    end_matches: list[PhraseMatch] = []

    start_kws = [kw for step in START_SEQUENCE for kw in step.keywords]
    end_kws = [kw for step in END_SEQUENCE for kw in step.keywords]

    for seg in segments:
        seg_lower = seg.text.lower()
        for phrase in start_kws:
            score = fuzz.partial_ratio(seg_lower, phrase)
            if score >= threshold:
                start_matches.append(PhraseMatch(
                    start=seg.start, end=seg.end, text=seg.text,
                    score=score, matched_phrase=phrase, role="start",
                ))
        for phrase in end_kws:
            score = fuzz.partial_ratio(seg_lower, phrase)
            if score >= threshold:
                end_matches.append(PhraseMatch(
                    start=seg.start, end=seg.end, text=seg.text,
                    score=score, matched_phrase=phrase, role="end",
                ))

    return start_matches, end_matches
