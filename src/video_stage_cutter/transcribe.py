"""Speech-to-text using faster-whisper."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
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
    data = [asdict(s) for s in segments]
    _write_json_atomic(path, data)
    log.debug("Saved transcript to %s", path)


def load_transcript(path: Path) -> list[TranscriptSegment]:
    """Load transcript segments from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return _segments_from_data(data)


def _segments_from_data(data: list[dict]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for seg in data:
        words = [
            WordInfo(start=w["start"], end=w["end"], word=w["word"], probability=w["probability"])
            for w in seg.get("words", [])
        ]
        segments.append(TranscriptSegment(
            start=seg["start"], end=seg["end"], text=seg["text"], words=words,
        ))
    return segments


def _write_json_atomic(path: Path, data: object) -> None:
    """Flush to a private sibling file, then replace without truncating a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def transcript_source(video_path: Path, model: str) -> dict:
    stat = video_path.stat()
    return {"path": str(video_path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "model": model}


def save_transcript_checkpoint(
    path: Path, segments: list[TranscriptSegment], source: dict, resume_from: float,
) -> None:
    _write_json_atomic(path, {
        "version": 1, "source": source, "resume_from": resume_from,
        "segments": [asdict(s) for s in segments],
    })


def load_transcript_checkpoint(path: Path, source: dict) -> tuple[list[TranscriptSegment], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["version"] != 1 or data["source"] != source:
        raise ValueError("Checkpoint source or Whisper model has changed")
    resume_from = float(data["resume_from"])
    if not math.isfinite(resume_from) or resume_from < 0:
        raise ValueError("Checkpoint contains an invalid resume position")
    return _segments_from_data(data["segments"]), resume_from
