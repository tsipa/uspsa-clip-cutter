"""Orchestrate per-video detection and cutting."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from video_stage_cutter.beep_detect import BeepCandidate, detect_beeps
from video_stage_cutter.cutter import cut_clip
from video_stage_cutter.ffmpeg_utils import extract_audio
from video_stage_cutter.manifest import ManifestRow
from video_stage_cutter.metadata import get_creation_time
from video_stage_cutter.phrase_detect import PhraseMatch, detect_phrases
from video_stage_cutter.transcribe import (
    TranscriptSegment,
    save_transcript,
    transcribe_audio,
)

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}


@dataclass
class ProcessingConfig:
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    accurate_cut: bool = True
    keep_wav: bool = False
    debug_dir: Path | None = None
    start_padding: float = 0.0
    end_padding: float = 2.0
    min_clip_length: float = 5.0
    max_clip_length: float = 600.0
    overwrite: bool = False
    dry_run: bool = False


@dataclass
class DetectionResult:
    transcript_path: str
    start_candidates: list[dict]
    end_candidates: list[dict]
    beep_candidates: list[dict]
    chosen_start: float | None
    chosen_end: float | None
    start_reason: str
    end_reason: str


def discover_videos(input_dir: Path) -> list[Path]:
    """Return sorted list of video files in *input_dir*."""
    videos = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    videos.sort(key=lambda p: p.name.lower())
    return videos


def _build_output_name(video_path: Path, creation_time_str: str) -> str:
    stem = video_path.stem
    ts = creation_time_str.replace(":", "-").replace(" ", "_")
    return f"{ts}__{stem}__stage_clip.mp4"


def process_single_video(
    video_path: Path,
    output_dir: Path,
    config: ProcessingConfig,
    whisper_model: object | None = None,
) -> tuple[ManifestRow, object | None]:
    """Process one video. Returns a manifest row and the (possibly cached) whisper model."""
    creation_dt = get_creation_time(video_path)
    creation_str = creation_dt.strftime("%Y-%m-%d_%H-%M-%S")

    debug_dir = config.debug_dir or output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = debug_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / f"{video_path.stem}.wav"

    row = ManifestRow(
        source_file=str(video_path),
        creation_time=creation_dt.isoformat(),
    )

    try:
        # --- extract audio ---
        log.info("Extracting audio from %s", video_path.name)
        extract_audio(video_path, wav_path)

        # --- transcribe ---
        segments, whisper_model = _transcribe(wav_path, config, whisper_model)

        transcript_path = debug_dir / f"{video_path.stem}_transcript.json"
        save_transcript(segments, transcript_path)

        # --- phrase detection ---
        start_matches, end_matches = detect_phrases(segments)

        # --- pick start ---
        chosen_start, start_reason, beep_candidates = _resolve_start(
            start_matches, wav_path, config,
        )

        # --- pick end ---
        chosen_end, end_reason = _resolve_end(end_matches, chosen_start)

        # --- apply padding ---
        if chosen_start is not None:
            chosen_start = max(0.0, chosen_start - config.start_padding)
        if chosen_end is not None:
            chosen_end = chosen_end + config.end_padding

        # --- save detection debug ---
        detection = DetectionResult(
            transcript_path=str(transcript_path),
            start_candidates=[asdict(m) for m in start_matches],
            end_candidates=[asdict(m) for m in end_matches],
            beep_candidates=[asdict(b) for b in beep_candidates],
            chosen_start=chosen_start,
            chosen_end=chosen_end,
            start_reason=start_reason,
            end_reason=end_reason,
        )
        det_path = debug_dir / f"{video_path.stem}_detection.json"
        det_path.write_text(
            json.dumps(asdict(detection), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # --- validate ---
        if chosen_start is None:
            raise ValueError("No start phrase detected")
        if chosen_end is None:
            raise ValueError("No end command detected")

        duration = chosen_end - chosen_start
        row.start_offset = f"{chosen_start:.3f}"
        row.end_offset = f"{chosen_end:.3f}"
        row.duration = f"{duration:.3f}"
        row.start_reason = start_reason
        row.end_reason = end_reason
        row.confidence = _compute_confidence(start_matches, end_matches, beep_candidates)

        # --- cut ---
        output_name = _build_output_name(video_path, creation_str)
        output_path = output_dir / output_name

        if output_path.exists() and not config.overwrite:
            row.status = "skipped"
            row.output_file = str(output_path)
            row.error_message = "Output already exists (use --overwrite)"
            log.warning("Skipping %s: output exists", video_path.name)
        elif config.dry_run:
            row.status = "dry_run"
            row.output_file = str(output_path)
            log.info("Dry run: would cut %s → %s", video_path.name, output_name)
        else:
            cut_clip(
                video_path,
                output_path,
                chosen_start,
                chosen_end,
                accurate=config.accurate_cut,
                min_clip_length=config.min_clip_length,
                max_clip_length=config.max_clip_length,
            )
            row.status = "ok"
            row.output_file = str(output_path)

    except Exception as exc:
        row.status = "failed"
        row.error_message = str(exc)
        log.error("Failed processing %s: %s", video_path.name, exc)

    finally:
        if not config.keep_wav and wav_path.exists():
            wav_path.unlink(missing_ok=True)

    return row, whisper_model


def _transcribe(
    wav_path: Path,
    config: ProcessingConfig,
    cached_model: object | None,
) -> tuple[list[TranscriptSegment], object]:
    """Transcribe, reusing the whisper model across videos."""
    from faster_whisper import WhisperModel

    if cached_model is None:
        log.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s) ...",
            config.model, config.device, config.compute_type,
        )
        cached_model = WhisperModel(
            config.model, device=config.device, compute_type=config.compute_type,
        )

    log.info("Transcribing %s ...", wav_path.name)
    segments_iter, _info = cached_model.transcribe(
        str(wav_path),
        beam_size=5,
        word_timestamps=True,
        language="en",
    )

    from video_stage_cutter.transcribe import WordInfo

    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        words = [
            WordInfo(start=w.start, end=w.end, word=w.word, probability=w.probability)
            for w in (seg.words or [])
        ]
        segments.append(TranscriptSegment(
            start=seg.start, end=seg.end, text=seg.text.strip(), words=words,
        ))

    log.info("Transcription complete: %d segments", len(segments))
    return segments, cached_model


def _resolve_start(
    start_matches: list[PhraseMatch],
    wav_path: Path,
    config: ProcessingConfig,
) -> tuple[float | None, str, list[BeepCandidate]]:
    """Pick the start time: prefer beep after 'stand by', else end of 'stand by'."""
    beep_candidates: list[BeepCandidate] = []

    standby = [m for m in start_matches if "stand by" in m.matched_phrase.lower()]
    ready = [m for m in start_matches if "ready" in m.matched_phrase.lower()]
    anchor = standby[0] if standby else (ready[0] if ready else None)

    if anchor is None:
        return None, "no_start_phrase", beep_candidates

    search_start = max(0.0, anchor.start - 1.0)
    search_end = anchor.end + 5.0

    beep_candidates = detect_beeps(wav_path, search_start, search_end)

    after_anchor = [b for b in beep_candidates if b.timestamp >= anchor.end - 0.2]

    if after_anchor:
        best = max(after_anchor[:3], key=lambda b: b.confidence)
        return best.timestamp, "beep_after_standby", beep_candidates

    return anchor.end, "standby_end_fallback", beep_candidates


def _resolve_end(
    end_matches: list[PhraseMatch],
    chosen_start: float | None,
) -> tuple[float | None, str]:
    """Pick the end time: prefer the longest matching phrase after the start."""
    if not end_matches:
        return None, "no_end_phrase"

    if chosen_start is not None:
        after_start = [m for m in end_matches if m.end > chosen_start]
        if after_start:
            best = max(after_start, key=lambda m: (len(m.matched_phrase), m.score))
            return best.end, f"matched:{best.matched_phrase}"

    best = max(end_matches, key=lambda m: (len(m.matched_phrase), m.score))
    return best.end, f"matched:{best.matched_phrase}"


def _compute_confidence(
    start_matches: list[PhraseMatch],
    end_matches: list[PhraseMatch],
    beep_candidates: list[BeepCandidate],
) -> str:
    scores = []
    if start_matches:
        scores.append(start_matches[0].score / 100.0)
    if end_matches:
        scores.append(max(m.score for m in end_matches) / 100.0)
    if beep_candidates:
        scores.append(min(1.0, max(b.confidence for b in beep_candidates)))

    if not scores:
        return "0.00"
    return f"{sum(scores) / len(scores):.2f}"


def run_batch(
    input_dir: Path,
    output_dir: Path,
    config: ProcessingConfig,
) -> list[ManifestRow]:
    """Process all videos in *input_dir*. Returns manifest rows."""
    from tqdm import tqdm

    videos = discover_videos(input_dir)
    if not videos:
        log.warning("No video files found in %s", input_dir)
        return []

    log.info("Found %d video(s) in %s", len(videos), input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    whisper_model = None

    for video_path in tqdm(videos, desc="Processing videos"):
        row, whisper_model = process_single_video(
            video_path, output_dir, config, whisper_model,
        )
        rows.append(row)

    return rows
