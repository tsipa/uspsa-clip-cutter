"""Orchestrate per-video detection and cutting with cross-file support."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from video_stage_cutter.beep_detect import BeepCandidate, detect_beeps
from video_stage_cutter.cutter import cut_clip
from video_stage_cutter.ffmpeg_utils import (
    concat_and_cut,
    extract_audio,
    get_duration,
)
from video_stage_cutter.manifest import ManifestRow
from video_stage_cutter.metadata import get_creation_time
from video_stage_cutter.phrase_detect import PhraseMatch, detect_phrases
from video_stage_cutter.transcribe import (
    TranscriptSegment,
    WordInfo,
    save_transcript,
)

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
MAX_STAGE_SECONDS = 300.0
FALLBACK_DURATION = 180.0


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
class FileDetection:
    """Per-file detection results collected in pass 1."""
    video_path: Path
    wav_path: Path
    duration: float
    creation_dt_iso: str
    creation_str: str
    segments: list[TranscriptSegment]
    start_matches: list[PhraseMatch]
    end_matches: list[PhraseMatch]
    beep_candidates: list[BeepCandidate]
    chosen_start: float | None
    start_reason: str
    chosen_end: float | None
    end_reason: str
    error: str | None = None


def discover_videos(input_dir: Path) -> list[Path]:
    """Return list of video files in *input_dir* sorted by creation time."""
    videos = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    videos.sort(key=lambda p: get_creation_time(p))
    return videos


def _build_output_name(
    video_path: Path,
    creation_time_str: str,
    tag: str = "",
) -> str:
    stem = video_path.stem
    ts = creation_time_str.replace(":", "-").replace(" ", "_")
    suffix = f"__{tag}" if tag else ""
    return f"{ts}__{stem}__stage_clip{suffix}.mp4"


# ---------------------------------------------------------------------------
# Pass 1: extract, transcribe, detect per file
# ---------------------------------------------------------------------------

def _detect_single(
    video_path: Path,
    config: ProcessingConfig,
    debug_dir: Path,
    whisper_model: object | None,
) -> tuple[FileDetection, object | None]:
    """Run extraction + transcription + detection for one file."""
    creation_dt = get_creation_time(video_path)
    creation_str = creation_dt.strftime("%Y-%m-%d_%H-%M-%S")

    wav_dir = debug_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / f"{video_path.stem}.wav"

    det = FileDetection(
        video_path=video_path,
        wav_path=wav_path,
        duration=0.0,
        creation_dt_iso=creation_dt.isoformat(),
        creation_str=creation_str,
        segments=[],
        start_matches=[],
        end_matches=[],
        beep_candidates=[],
        chosen_start=None,
        start_reason="no_start_phrase",
        chosen_end=None,
        end_reason="no_end_phrase",
    )

    try:
        det.duration = get_duration(video_path)
        log.info("=== %s (%.1fs) ===", video_path.name, det.duration)

        log.info("Extracting audio ...")
        extract_audio(video_path, wav_path)

        det.segments, whisper_model = _transcribe(wav_path, config, whisper_model)
        log.info("Transcript: %d segments, full text preview: %.200s",
                 len(det.segments),
                 " | ".join(s.text for s in det.segments[:10]))

        transcript_path = debug_dir / f"{video_path.stem}_transcript.json"
        save_transcript(det.segments, transcript_path)

        log.info("Running phrase detection ...")
        det.start_matches, det.end_matches = detect_phrases(det.segments)

        log.info("Resolving start ...")
        det.chosen_start, det.start_reason, det.beep_candidates = _resolve_start(
            det.start_matches, wav_path,
        )

        log.info("Resolving end ...")
        det.chosen_end, det.end_reason = _resolve_end(
            det.end_matches, det.chosen_start,
        )

        log.info(
            "Detection result for %s: start=%.3fs (%s), end=%s (%s)",
            video_path.name,
            det.chosen_start if det.chosen_start is not None else -1,
            det.start_reason,
            f"{det.chosen_end:.3f}s" if det.chosen_end is not None else "NONE",
            det.end_reason,
        )

    except Exception as exc:
        det.error = str(exc)
        log.error("Detection failed for %s: %s", video_path.name, exc)

    return det, whisper_model


# ---------------------------------------------------------------------------
# Pass 2: resolve cross-file stages and cut
# ---------------------------------------------------------------------------

def _resolve_and_cut(
    detections: list[FileDetection],
    output_dir: Path,
    config: ProcessingConfig,
    debug_dir: Path,
) -> list[ManifestRow]:
    """Match starts with ends across adjacent files, cut clips."""
    rows: list[ManifestRow] = []
    consumed: set[int] = set()

    for i, det in enumerate(detections):
        if i in consumed:
            continue

        if det.error:
            rows.append(_error_row(det, det.error))
            consumed.add(i)
            continue

        has_start = det.chosen_start is not None
        has_end = det.chosen_end is not None

        # --- normal: both start and end in same file ---
        if has_start and has_end:
            log.info(
                "[%s] Both start (%.2fs) and end (%.2fs) in same file — normal cut",
                det.video_path.name, det.chosen_start, det.chosen_end,
            )
            row = _cut_single(det, output_dir, config, debug_dir)
            rows.append(row)
            consumed.add(i)
            continue

        # --- start found, no end: look in next file ---
        if has_start and not has_end:
            log.warning(
                "[%s] Start at %.2fs but no end — checking next file ...",
                det.video_path.name, det.chosen_start,
            )
            next_i = i + 1
            if next_i < len(detections) and next_i not in consumed:
                next_det = detections[next_i]
                if next_det.error is None and next_det.chosen_end is not None:
                    end_in_next = next_det.chosen_end
                    combined_end = det.duration + end_in_next
                    combined_dur = combined_end - det.chosen_start
                    if combined_dur <= MAX_STAGE_SECONDS:
                        log.info(
                            "[%s+%s] Found end in next file at %.2fs "
                            "(combined offset %.2fs, stage duration %.1fs) — cross-file cut",
                            det.video_path.name, next_det.video_path.name,
                            end_in_next, combined_end, combined_dur,
                        )
                        row = _cut_cross_file(
                            det, next_det, det.chosen_start, combined_end,
                            output_dir, config, debug_dir,
                            start_reason=det.start_reason,
                            end_reason=next_det.end_reason,
                        )
                        rows.append(row)
                        consumed.add(i)
                        consumed.add(next_i)
                        continue
                    else:
                        log.warning(
                            "[%s+%s] End in next file would make stage %.1fs (>%.0fs max) — ignoring",
                            det.video_path.name, next_det.video_path.name,
                            combined_dur, MAX_STAGE_SECONDS,
                        )
                else:
                    log.warning(
                        "[%s] Next file %s has no usable end (error=%s, end=%s)",
                        det.video_path.name, next_det.video_path.name,
                        next_det.error, next_det.chosen_end,
                    )
            else:
                log.warning("[%s] No next file available to check", det.video_path.name)

            log.error(
                "[%s] FALLBACK: no end found anywhere — recording %.0fs after beep at %.2fs",
                det.video_path.name, FALLBACK_DURATION, det.chosen_start,
            )
            row = _cut_single_fallback_no_end(det, output_dir, config, debug_dir)
            rows.append(row)
            consumed.add(i)
            continue

        # --- end found, no start: 3 min before end ---
        if has_end and not has_start:
            log.error(
                "[%s] FALLBACK: end at %.2fs but no start — recording %.0fs before end",
                det.video_path.name, det.chosen_end, FALLBACK_DURATION,
            )
            row = _cut_single_fallback_no_start(det, output_dir, config, debug_dir)
            rows.append(row)
            consumed.add(i)
            continue

        # --- neither start nor end ---
        log.warning("[%s] Nothing detected — no start, no end, skipping", det.video_path.name)
        rows.append(_error_row(det, "No start phrase and no end command detected"))
        consumed.add(i)

    return rows


# ---------------------------------------------------------------------------
# Cutting helpers
# ---------------------------------------------------------------------------

def _cut_single(
    det: FileDetection,
    output_dir: Path,
    config: ProcessingConfig,
    debug_dir: Path,
) -> ManifestRow:
    start = max(0.0, det.chosen_start - config.start_padding)
    end = det.chosen_end + config.end_padding
    end = min(end, det.duration)
    name = _build_output_name(det.video_path, det.creation_str)
    return _do_cut(
        sources=[det.video_path],
        start=start, end=end,
        output_name=name,
        det=det, output_dir=output_dir,
        config=config, debug_dir=debug_dir,
        start_reason=det.start_reason,
        end_reason=det.end_reason,
    )


def _cut_single_fallback_no_end(
    det: FileDetection,
    output_dir: Path,
    config: ProcessingConfig,
    debug_dir: Path,
) -> ManifestRow:
    start = max(0.0, det.chosen_start - config.start_padding)
    end = min(start + FALLBACK_DURATION, det.duration)
    name = _build_output_name(det.video_path, det.creation_str, tag="no_end")
    return _do_cut(
        sources=[det.video_path],
        start=start, end=end,
        output_name=name,
        det=det, output_dir=output_dir,
        config=config, debug_dir=debug_dir,
        start_reason=det.start_reason,
        end_reason="fallback_3min_no_end",
    )


def _cut_single_fallback_no_start(
    det: FileDetection,
    output_dir: Path,
    config: ProcessingConfig,
    debug_dir: Path,
) -> ManifestRow:
    end = det.chosen_end + config.end_padding
    end = min(end, det.duration)
    start = max(0.0, end - FALLBACK_DURATION)
    name = _build_output_name(det.video_path, det.creation_str, tag="no_start")
    return _do_cut(
        sources=[det.video_path],
        start=start, end=end,
        output_name=name,
        det=det, output_dir=output_dir,
        config=config, debug_dir=debug_dir,
        start_reason="fallback_3min_no_start",
        end_reason=det.end_reason,
    )


def _cut_cross_file(
    det1: FileDetection,
    det2: FileDetection,
    start: float,
    combined_end: float,
    output_dir: Path,
    config: ProcessingConfig,
    debug_dir: Path,
    start_reason: str,
    end_reason: str,
) -> ManifestRow:
    adj_start = max(0.0, start - config.start_padding)
    adj_end = combined_end + config.end_padding
    name = _build_output_name(
        det1.video_path, det1.creation_str, tag="split",
    )
    log.info(
        "Cross-file cut: %s + %s (%.2f–%.2f on combined timeline)",
        det1.video_path.name, det2.video_path.name, adj_start, adj_end,
    )
    return _do_cut(
        sources=[det1.video_path, det2.video_path],
        start=adj_start, end=adj_end,
        output_name=name,
        det=det1, output_dir=output_dir,
        config=config, debug_dir=debug_dir,
        start_reason=start_reason,
        end_reason=f"cross_file:{end_reason}",
        source_override=f"{det1.video_path}+{det2.video_path}",
    )


def _do_cut(
    sources: list[Path],
    start: float,
    end: float,
    output_name: str,
    det: FileDetection,
    output_dir: Path,
    config: ProcessingConfig,
    debug_dir: Path,
    start_reason: str,
    end_reason: str,
    source_override: str | None = None,
) -> ManifestRow:
    duration = end - start
    output_path = output_dir / output_name

    row = ManifestRow(
        source_file=source_override or str(det.video_path),
        creation_time=det.creation_dt_iso,
        duration=f"{duration:.3f}",
        start_offset=f"{start:.3f}",
        end_offset=f"{end:.3f}",
        start_reason=start_reason,
        end_reason=end_reason,
        confidence=_compute_confidence(
            det.start_matches, det.end_matches, det.beep_candidates,
        ),
    )

    _save_detection_debug(det, debug_dir, start, end, start_reason, end_reason)

    if output_path.exists() and not config.overwrite:
        row.status = "skipped"
        row.output_file = str(output_path)
        row.error_message = "Output already exists (use --overwrite)"
        return row

    if config.dry_run:
        row.status = "dry_run"
        row.output_file = str(output_path)
        return row

    if duration < config.min_clip_length:
        row.status = "failed"
        row.error_message = f"Clip too short: {duration:.1f}s < {config.min_clip_length:.1f}s"
        return row
    if duration > config.max_clip_length:
        row.status = "failed"
        row.error_message = f"Clip too long: {duration:.1f}s > {config.max_clip_length:.1f}s"
        return row

    try:
        if len(sources) == 1:
            cut_clip(
                sources[0], output_path, start, end,
                accurate=config.accurate_cut,
                min_clip_length=config.min_clip_length,
                max_clip_length=config.max_clip_length,
            )
        else:
            concat_and_cut(
                sources, output_path, start, end,
                accurate=config.accurate_cut,
            )
        row.status = "ok"
        row.output_file = str(output_path)
    except Exception as exc:
        row.status = "failed"
        row.error_message = str(exc)
        log.error("Cut failed for %s: %s", output_name, exc)

    return row


def _error_row(det: FileDetection, msg: str) -> ManifestRow:
    return ManifestRow(
        source_file=str(det.video_path),
        creation_time=det.creation_dt_iso,
        status="failed",
        error_message=msg,
    )


# ---------------------------------------------------------------------------
# Detection internals
# ---------------------------------------------------------------------------

def _transcribe(
    wav_path: Path,
    config: ProcessingConfig,
    cached_model: object | None,
) -> tuple[list[TranscriptSegment], object]:
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
        str(wav_path), beam_size=5, word_timestamps=True, language="en",
    )

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
) -> tuple[float | None, str, list[BeepCandidate]]:
    beep_candidates: list[BeepCandidate] = []

    standby = [m for m in start_matches if "stand by" in m.matched_phrase.lower()]
    ready = [m for m in start_matches if "ready" in m.matched_phrase.lower()]

    if standby:
        anchor = standby[0]
        log.info("Start anchor: 'stand by' at %.2f–%.2fs (score=%.0f)", anchor.start, anchor.end, anchor.score)
    elif ready:
        anchor = ready[0]
        log.info("Start anchor: 'are you ready' at %.2f–%.2fs (score=%.0f), no 'stand by' found", anchor.start, anchor.end, anchor.score)
    else:
        log.warning("No start anchor found — neither 'stand by' nor 'are you ready' in transcript")
        return None, "no_start_phrase", beep_candidates

    search_start = anchor.end + 1.0
    search_end = anchor.end + 10.0
    log.info("Searching for beep in window %.2f–%.2fs (1–10s after anchor end)", search_start, search_end)

    beep_candidates = detect_beeps(wav_path, search_start, search_end)

    if beep_candidates:
        best = max(beep_candidates[:3], key=lambda b: b.confidence)
        log.info(
            "Using beep at %.3fs as start (confidence=%.3f, %.2fs after anchor end)",
            best.timestamp, best.confidence, best.timestamp - anchor.end,
        )
        return best.timestamp, "beep_after_standby", beep_candidates

    log.warning(
        "No beep found after anchor — falling back to end of '%s' at %.2fs",
        anchor.matched_phrase, anchor.end,
    )
    return anchor.end, "standby_end_fallback", beep_candidates


def _resolve_end(
    end_matches: list[PhraseMatch],
    chosen_start: float | None,
) -> tuple[float | None, str]:
    if not end_matches:
        log.warning("No end command candidates found in transcript")
        return None, "no_end_phrase"

    if chosen_start is not None:
        after_start = [m for m in end_matches if m.end > chosen_start]
        if after_start:
            best = max(after_start, key=lambda m: (len(m.matched_phrase), m.score))
            log.info(
                "End command: '%s' at %.2f–%.2fs (score=%.0f, matched '%s')",
                best.text, best.start, best.end, best.score, best.matched_phrase,
            )
            return best.end, f"matched:{best.matched_phrase}"
        log.warning(
            "All %d end candidates are before start (%.2fs), ignoring them",
            len(end_matches), chosen_start,
        )
        return None, "no_end_phrase"

    best = max(end_matches, key=lambda m: (len(m.matched_phrase), m.score))
    log.info(
        "End command (no start context): '%s' at %.2f–%.2fs (score=%.0f, matched '%s')",
        best.text, best.start, best.end, best.score, best.matched_phrase,
    )
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


def _save_detection_debug(
    det: FileDetection,
    debug_dir: Path,
    chosen_start: float,
    chosen_end: float,
    start_reason: str,
    end_reason: str,
) -> None:
    data = {
        "video": str(det.video_path),
        "duration": det.duration,
        "start_candidates": [asdict(m) for m in det.start_matches],
        "end_candidates": [asdict(m) for m in det.end_matches],
        "beep_candidates": [asdict(b) for b in det.beep_candidates],
        "chosen_start": chosen_start,
        "chosen_end": chosen_end,
        "start_reason": start_reason,
        "end_reason": end_reason,
    }
    path = debug_dir / f"{det.video_path.stem}_detection.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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

    debug_dir = config.debug_dir or output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # --- pass 1: detect everything ---
    detections: list[FileDetection] = []
    whisper_model = None

    for video_path in tqdm(videos, desc="Pass 1: detecting"):
        det, whisper_model = _detect_single(
            video_path, config, debug_dir, whisper_model,
        )
        detections.append(det)

    # --- pass 2: resolve cross-file + cut ---
    log.info("Pass 2: resolving stages and cutting clips ...")
    rows = _resolve_and_cut(detections, output_dir, config, debug_dir)

    # --- cleanup wav ---
    if not config.keep_wav:
        for det in detections:
            if det.wav_path.exists():
                det.wav_path.unlink(missing_ok=True)

    return rows
