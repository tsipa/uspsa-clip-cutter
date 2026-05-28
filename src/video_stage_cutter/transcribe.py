"""Speech-to-text using faster-whisper."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class WordInfo:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[WordInfo]



def save_transcript(segments: list[TranscriptSegment], path: Path) -> None:
    """Write transcript segments as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(s) for s in segments]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.debug("Saved transcript to %s", path)
