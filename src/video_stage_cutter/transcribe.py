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


def transcribe_audio(
    audio_path: Path,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[TranscriptSegment]:
    """Transcribe *audio_path* and return a list of segments with word-level timing."""
    from faster_whisper import WhisperModel

    log.info("Loading Whisper model '%s' (device=%s, compute_type=%s) ...", model_name, device, compute_type)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    log.info("Transcribing %s ...", audio_path.name)
    segments_iter, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        language="en",
        initial_prompt=(
            "USPSA shooting match. Range officer commands: "
            "Shooter make ready. Are you ready? Stand by. "
            "If clear, hammer down and holster. "
            "If finished, unload and show clear. Range is clear."
        ),
        vad_filter=True,
    )

    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        words = [
            WordInfo(
                start=w.start,
                end=w.end,
                word=w.word,
                probability=w.probability,
            )
            for w in (seg.words or [])
        ]
        segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=words,
        ))

    log.info("Transcription complete: %d segments", len(segments))
    return segments


def save_transcript(segments: list[TranscriptSegment], path: Path) -> None:
    """Write transcript segments as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(s) for s in segments]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.debug("Saved transcript to %s", path)
