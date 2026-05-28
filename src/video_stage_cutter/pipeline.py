"""Anchor-based pipeline: collect all events globally, then assemble stages."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from video_stage_cutter.beep_detect import detect_beeps, detect_gunshots
from video_stage_cutter.cutter import cut_clip
from video_stage_cutter.ffmpeg_utils import (
    concat_and_cut,
    extract_audio,
    get_duration,
)
from video_stage_cutter.manifest import ManifestRow
from video_stage_cutter.metadata import (
    filename_sort_key,
    get_creation_time,
    get_creation_time_or_mtime,
)
from video_stage_cutter.phrase_detect import detect_phrases
from video_stage_cutter.transcribe import (
    TranscriptSegment,
    WordInfo,
    save_transcript,
)

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
MAX_STAGE_SECONDS = 300.0
FALLBACK_DURATION = 180.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ProcessingConfig:
    model: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "int8"
    accurate_cut: bool = True
    keep_wav: bool = False
    debug_dir: Path | None = None
    start_padding: float = 10.0
    end_padding: float = 10.0
    min_clip_length: float = 5.0
    max_clip_length: float = 600.0
    overwrite: bool = False
    dry_run: bool = False
    phrase_threshold: float = 70.0
    beep_search_before: float = 0.25
    beep_search_after: float = 10.0
    workers: int = 1


# ---------------------------------------------------------------------------
# Anchor: a single detected event on the global timeline
# ---------------------------------------------------------------------------

@dataclass
class Anchor:
    kind: str           # "ready", "standby", "beep", "gunshot", "end_command"
    abs_time: float     # epoch seconds (creation_time + file offset)
    file_idx: int       # index in the sorted file list
    file_offset: float  # seconds into this file
    text: str           # matched text or description
    score: float        # confidence / fuzzy score
    end_offset: float = 0.0  # end of the phrase in file-local time

    def __repr__(self) -> str:
        return f"<{self.kind} t={self.abs_time:.2f} off={self.file_offset:.2f} '{self.text}' score={self.score:.0f}>"


# ---------------------------------------------------------------------------
# Beep search debug record
# ---------------------------------------------------------------------------

@dataclass
class BeepSearchRecord:
    anchor_offset: float
    anchor_kind: str
    search_start: float
    search_end: float
    candidates: list[dict] = field(default_factory=list)
    chosen_timestamp: float | None = None
    chosen_reason: str = ""


# ---------------------------------------------------------------------------
# Stage: assembled from anchors
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    ready: Anchor | None = None
    standby: Anchor | None = None
    beep: Anchor | None = None
    gunshots: list[Anchor] = field(default_factory=list)
    end_command: Anchor | None = None

    clip_start: float = 0.0   # abs_time
    clip_end: float = 0.0     # abs_time
    start_reason: str = ""
    end_reason: str = ""
    complete: bool = False

    # set during overlap trimming for fallback stages
    original_clip_start: float | None = None
    original_clip_end: float | None = None
    trimmed: bool = False
    trimmed_by: str = ""

    @property
    def duration(self) -> float:
        return self.clip_end - self.clip_start


# ---------------------------------------------------------------------------
# Per-file info
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    path: Path
    wav_path: Path
    duration: float
    creation_epoch: float
    creation_str: str
    creation_iso: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    beep_searches: list[BeepSearchRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pass 1: collect anchors
# ---------------------------------------------------------------------------

def discover_videos(input_dir: Path) -> list[Path]:
    videos = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    has_metadata = all(get_creation_time(v) is not None for v in videos)
    if has_metadata:
        videos.sort(key=lambda p: get_creation_time(p))
        log.info("Sorted %d videos by creation_time metadata", len(videos))
    else:
        videos.sort(key=filename_sort_key)
        log.info("Some files lack creation_time -- sorted %d videos by filename (GoPro-aware)", len(videos))
    return videos


def _collect_anchors_for_file(
    file_info: FileInfo,
    file_idx: int,
    config: ProcessingConfig,
    debug_dir: Path,
    whisper_model: object | None,
) -> tuple[list[Anchor], object | None]:
    """Extract all anchors from one file."""
    anchors: list[Anchor] = []
    fi = file_info
    epoch = fi.creation_epoch

    log.info("=" * 60)
    log.info("FILE [%d]: %s (%.1fs, created %s)", file_idx, fi.path.name, fi.duration, fi.creation_str)

    log.info("  Extracting audio ...")
    extract_audio(fi.path, fi.wav_path)

    fi.segments, whisper_model = _transcribe(fi.wav_path, config, whisper_model)
    log.info("  Transcript (%d segments):", len(fi.segments))
    for seg in fi.segments:
        log.info("    [%.2f-%.2f] %s", seg.start, seg.end, seg.text)

    save_transcript(fi.segments, debug_dir / f"{fi.path.stem}_transcript.json")

    start_matches, end_matches = detect_phrases(fi.segments, threshold=config.phrase_threshold)

    for m in start_matches:
        kind = "standby" if "stand by" in m.matched_phrase.lower() else "ready"
        anchors.append(Anchor(
            kind=kind, abs_time=epoch + m.start, file_idx=file_idx,
            file_offset=m.start, text=m.text, score=m.score,
            end_offset=m.end,
        ))
        log.info("  ANCHOR %s: %.2fs '%s' (matched '%s', score=%.0f)",
                 kind.upper(), m.start, m.text, m.matched_phrase, m.score)

    for m in end_matches:
        anchors.append(Anchor(
            kind="end_command", abs_time=epoch + m.start, file_idx=file_idx,
            file_offset=m.start, text=m.text, score=m.score,
            end_offset=m.end,
        ))
        log.info("  ANCHOR END_COMMAND: %.2fs '%s' (matched '%s', score=%.0f)",
                 m.start, m.text, m.matched_phrase, m.score)

    # --- beep detection: search around standby, or make_ready with wider window ---
    standby_anchors = [a for a in anchors if a.kind == "standby"]
    ready_anchors = [a for a in anchors if a.kind == "ready"]

    # if we have standby, search near standby (tight window)
    # if no standby but have ready/make_ready, search with wider window (up to 90s)
    # beep search always stops at the first end_command after the anchor
    end_command_anchors = sorted(
        [a for a in anchors if a.kind == "end_command"],
        key=lambda a: a.file_offset,
    )

    beep_search_sources: list[tuple[Anchor, float, float]] = []
    for sb in standby_anchors:
        beep_search_sources.append((sb, config.beep_search_before, config.beep_search_after))

    if not standby_anchors and ready_anchors:
        for ra in ready_anchors:
            beep_search_sources.append((ra, config.beep_search_before, 90.0))
            log.info("  No standby found, using ready anchor at %.2fs with wider beep search window", ra.file_offset)

    for anchor, search_before, search_after in beep_search_sources:
        search_start = max(0.0, anchor.end_offset - search_before)
        search_end = anchor.end_offset + search_after

        # clamp search_end to first end_command after anchor
        for ec in end_command_anchors:
            if ec.file_offset > anchor.end_offset:
                if ec.file_offset < search_end:
                    log.info("  Clamping beep search end from %.2fs to %.2fs (end_command at %.2fs)",
                             search_end, ec.file_offset, ec.file_offset)
                    search_end = ec.file_offset
                break

        log.info("  Searching beep around %s at %.2fs, window %.2f-%.2fs",
                 anchor.kind, anchor.file_offset, search_start, search_end)

        beeps = detect_beeps(fi.wav_path, search_start, search_end)

        record = BeepSearchRecord(
            anchor_offset=anchor.file_offset,
            anchor_kind=anchor.kind,
            search_start=search_start,
            search_end=search_end,
        )

        if beeps:
            best = max(beeps, key=lambda b: (b.tonality, b.band_energy))
            log.info("  Accepted beep candidates:")
            for bc in beeps:
                chosen = bc is best
                marker = " <-- CHOSEN" if chosen else ""
                log.info("    t=%.3fs energy=%.1f tonality=%.3f bb_ratio=%.3f duration=%.0fms%s",
                         bc.timestamp, bc.band_energy, bc.tonality, bc.broadband_ratio,
                         bc.duration_ms, marker)
                record.candidates.append({
                    "timestamp": bc.timestamp,
                    "band_energy": bc.band_energy,
                    "tonality": bc.tonality,
                    "broadband_ratio": bc.broadband_ratio,
                    "spectral_flatness": bc.spectral_flatness,
                    "duration_ms": bc.duration_ms,
                    "neighbors_1s": bc.neighbors_1s,
                    "chosen": chosen,
                })

            record.chosen_timestamp = best.timestamp
            record.chosen_reason = (
                f"best tonality={best.tonality:.3f} energy={best.band_energy:.1f} "
                f"duration={best.duration_ms:.0f}ms"
            )

            anchors.append(Anchor(
                kind="beep", abs_time=epoch + best.timestamp, file_idx=file_idx,
                file_offset=best.timestamp,
                text=f"timer_beep (tonality={best.tonality:.2f} energy={best.band_energy:.0f} {best.duration_ms:.0f}ms)",
                score=min(100.0, best.tonality * 200),
            ))
            log.info("  ANCHOR BEEP: %.3fs (tonality=%.3f, energy=%.1f, %.2fs after anchor end)",
                     best.timestamp, best.tonality, best.band_energy, best.timestamp - anchor.end_offset)
        else:
            record.chosen_reason = "no_candidates"
            log.warning("  No beep found around %s at %.2fs", anchor.kind, anchor.file_offset)

        fi.beep_searches.append(record)

    # --- gunshot detection: full file ---
    log.info("  Detecting gunshots ...")
    gunshots = detect_gunshots(fi.wav_path)
    for gs in gunshots:
        anchors.append(Anchor(
            kind="gunshot", abs_time=epoch + gs.timestamp, file_idx=file_idx,
            file_offset=gs.timestamp, text="gunshot", score=gs.confidence * 100,
        ))
    if gunshots:
        log.info("  ANCHOR GUNSHOTS: %d detected (first at %.2fs, last at %.2fs)",
                 len(gunshots), gunshots[0].timestamp, gunshots[-1].timestamp)
    else:
        log.info("  No gunshots detected")

    return anchors, whisper_model


# ---------------------------------------------------------------------------
# Pass 2: assemble stages from anchors (confirmed first, then fallbacks)
# ---------------------------------------------------------------------------

def _dedup_anchors(anchors: list[Anchor], time_tolerance: float = 10.0) -> list[Anchor]:
    """Within each *time_tolerance* window, keep only the best anchor per kind.

    Best = longest text (longest phrase match), then highest score.
    Beeps and gunshots are not deduped (they use different logic).
    """
    dedup_kinds = {"ready", "standby", "end_command"}
    passthrough = [a for a in anchors if a.kind not in dedup_kinds]
    to_dedup = [a for a in anchors if a.kind in dedup_kinds]

    result: list[Anchor] = []
    for kind in dedup_kinds:
        kind_anchors = sorted([a for a in to_dedup if a.kind == kind], key=lambda a: a.abs_time)
        if not kind_anchors:
            continue
        cluster: list[Anchor] = [kind_anchors[0]]
        for a in kind_anchors[1:]:
            if a.abs_time - cluster[0].abs_time < time_tolerance:
                cluster.append(a)
            else:
                best = max(cluster, key=lambda x: (len(x.text), x.score))
                result.append(best)
                if len(cluster) > 1:
                    log.info("  Anchor dedup: %s — kept '%s' (score=%.0f), dropped %d others within %.1fs",
                             kind, best.text, best.score, len(cluster) - 1, time_tolerance)
                cluster = [a]
        best = max(cluster, key=lambda x: (len(x.text), x.score))
        result.append(best)
        if len(cluster) > 1:
            log.info("  Anchor dedup: %s — kept '%s' (score=%.0f), dropped %d others within %.1fs",
                     kind, best.text, best.score, len(cluster) - 1, time_tolerance)

    result.extend(passthrough)
    result.sort(key=lambda a: a.abs_time)
    return result


def _assemble_stages(anchors: list[Anchor], min_clip_length: float = 5.0) -> list[Stage]:
    """Walk the sorted anchor timeline and group into stages.

    Confirmed stages (beep + end_command) are built first.
    Fallback stages are built after, trimmed so they don't overlap confirmed ones.
    """
    anchors.sort(key=lambda a: a.abs_time)

    # dedup anchors: within 10s, keep only the best per kind
    anchors = _dedup_anchors(anchors, time_tolerance=10.0)

    log.info("=" * 60)
    log.info("ASSEMBLY: %d anchors on global timeline (after dedup)", len(anchors))
    for a in anchors:
        log.info("  %.2f [file %d @ %.2fs] %s '%s' score=%.0f",
                 a.abs_time, a.file_idx, a.file_offset, a.kind.upper(), a.text, a.score)

    used_beeps: set[int] = set()
    used_ends: set[int] = set()
    confirmed: list[Stage] = []
    fallback_no_end: list[Stage] = []

    beep_indices = [i for i, a in enumerate(anchors) if a.kind == "beep"]

    # --- first pass: build confirmed stages (beep + end_command) ---
    for bi in beep_indices:
        beep = anchors[bi]
        end_idx = None

        for j in range(bi + 1, len(anchors)):
            a = anchors[j]
            if a.abs_time - beep.abs_time > MAX_STAGE_SECONDS:
                break
            if a.kind == "end_command" and j not in used_ends:
                end_idx = j
                break

        if end_idx is not None:
            stage = _build_stage_from_beep(anchors, bi, end_idx)
            confirmed.append(stage)
            used_beeps.add(bi)
            used_ends.add(end_idx)
            # also mark nearby end_commands as used (dedup "hammer down" vs
            # "hammer down and holster" from the same moment)
            end_time = anchors[end_idx].abs_time
            for j, a in enumerate(anchors):
                if a.kind == "end_command" and j not in used_ends and abs(a.abs_time - end_time) < 5.0:
                    used_ends.add(j)
            _log_stage(stage, len(confirmed), "CONFIRMED")

    # --- second pass: beeps without end → fallback no-end ---
    for bi in beep_indices:
        if bi in used_beeps:
            continue
        beep = anchors[bi]
        stage = _build_stage_from_beep(anchors, bi, end_idx=None)
        stage.clip_end = beep.abs_time + FALLBACK_DURATION
        stage.end_reason = "fallback_3min_no_end"
        stage.complete = False
        fallback_no_end.append(stage)
        used_beeps.add(bi)

    # --- third pass: standby/ready without beep + end_command → confirmed ---
    # If beep wasn't detected but we have a start phrase + end command,
    # use the start phrase end as the stage start.
    used_starts: set[int] = set()
    start_indices = [i for i, a in enumerate(anchors) if a.kind in ("standby", "ready")]
    for si in start_indices:
        start_a = anchors[si]
        # skip if this start is already associated with a confirmed stage
        already_used = False
        for stage in confirmed:
            if stage.standby and abs(stage.standby.abs_time - start_a.abs_time) < 1.0:
                already_used = True
                break
            if stage.ready and abs(stage.ready.abs_time - start_a.abs_time) < 1.0:
                already_used = True
                break
        if already_used:
            continue

        end_idx = None
        for j in range(si + 1, len(anchors)):
            a = anchors[j]
            if a.abs_time - start_a.abs_time > MAX_STAGE_SECONDS:
                break
            if a.kind == "end_command" and j not in used_ends:
                end_idx = j
                break

        if end_idx is not None:
            end_a = anchors[end_idx]
            stage = Stage(end_command=end_a)
            if start_a.kind == "standby":
                stage.standby = start_a
            else:
                stage.ready = start_a
            stage.clip_start = start_a.end_offset + (start_a.abs_time - start_a.file_offset)
            stage.clip_end = end_a.abs_time + (end_a.end_offset - end_a.file_offset)
            stage.start_reason = f"{start_a.kind}_no_beep"
            stage.end_reason = f"matched:{end_a.text}"
            stage.complete = True
            for j in range(si + 1, end_idx):
                if anchors[j].kind == "gunshot":
                    stage.gunshots.append(anchors[j])
            confirmed.append(stage)
            used_starts.add(si)
            used_ends.add(end_idx)
            end_time = end_a.abs_time
            for j, a in enumerate(anchors):
                if a.kind == "end_command" and j not in used_ends and abs(a.abs_time - end_time) < 5.0:
                    used_ends.add(j)
            _log_stage(stage, len(confirmed), "CONFIRMED (no beep)")

    # --- fourth pass: start without beep or end → fallback 3min from start ---
    fallback_start_only: list[Stage] = []
    for si in start_indices:
        if si in used_starts:
            continue
        start_a = anchors[si]
        already_used = False
        for stage in confirmed + fallback_no_end:
            if stage.standby and abs(stage.standby.abs_time - start_a.abs_time) < 1.0:
                already_used = True
                break
            if stage.ready and abs(stage.ready.abs_time - start_a.abs_time) < 1.0:
                already_used = True
                break
            if stage.beep and abs(stage.beep.abs_time - start_a.abs_time) < 30.0:
                already_used = True
                break
        if already_used:
            continue

        stage = Stage()
        if start_a.kind == "standby":
            stage.standby = start_a
        else:
            stage.ready = start_a
        epoch_offset = start_a.abs_time - start_a.file_offset
        stage.clip_start = epoch_offset + start_a.end_offset
        stage.clip_end = stage.clip_start + FALLBACK_DURATION
        stage.start_reason = f"{start_a.kind}_no_beep"
        stage.end_reason = "fallback_3min_no_end"
        stage.complete = False
        for ga in anchors:
            if ga.kind == "gunshot" and stage.clip_start <= ga.abs_time <= stage.clip_end:
                stage.gunshots.append(ga)
        fallback_start_only.append(stage)
        used_starts.add(si)

    # --- fifth pass: orphan end_commands → fallback no-start ---
    # deduplicate orphan ends by time proximity: keep only the best-scoring
    # end_command within each 5s window
    fallback_no_start: list[Stage] = []
    orphan_ends = [
        (i, a) for i, a in enumerate(anchors)
        if a.kind == "end_command" and i not in used_ends
    ]
    orphan_used: set[int] = set()
    for idx, a in orphan_ends:
        if idx in orphan_used:
            continue
        # find all orphan ends within 5s of this one, pick best
        cluster = [(idx, a)]
        for idx2, a2 in orphan_ends:
            if idx2 != idx and idx2 not in orphan_used and abs(a2.abs_time - a.abs_time) < 5.0:
                cluster.append((idx2, a2))
        best_idx, best_a = max(cluster, key=lambda x: (len(x[1].text), x[1].score))
        for ci, _ in cluster:
            orphan_used.add(ci)

        stage = Stage(end_command=best_a)
        stage.clip_end = best_a.abs_time + (best_a.end_offset - best_a.file_offset)
        stage.clip_start = stage.clip_end - FALLBACK_DURATION
        stage.start_reason = "fallback_3min_no_start"
        stage.end_reason = f"matched:{best_a.text}"
        stage.complete = False
        for ga in anchors:
            if ga.kind == "gunshot" and stage.clip_start <= ga.abs_time <= stage.clip_end:
                stage.gunshots.append(ga)
        fallback_no_start.append(stage)

    # --- trim fallbacks against confirmed intervals ---
    confirmed_intervals = [(s.clip_start, s.clip_end) for s in confirmed]

    all_fallbacks = fallback_no_end + fallback_start_only + fallback_no_start
    trimmed_fallbacks: list[Stage] = []

    for stage in all_fallbacks:
        trimmed = _trim_fallback(stage, confirmed_intervals, min_clip_length)
        if trimmed is not None:
            trimmed_fallbacks.append(trimmed)
            _log_stage(trimmed, len(confirmed) + len(trimmed_fallbacks), "FALLBACK")
        else:
            log.warning("  Fallback stage skipped: overlaps confirmed stage and remaining interval too short")

    stages = confirmed + trimmed_fallbacks
    stages.sort(key=lambda s: s.clip_start)

    # --- final dedup: remove stages that substantially overlap a better stage ---
    stages = _dedup_overlapping_stages(stages)

    return stages


def _build_stage_from_beep(
    anchors: list[Anchor],
    beep_idx: int,
    end_idx: int | None,
) -> Stage:
    """Build a stage starting from a beep anchor."""
    beep = anchors[beep_idx]
    stage = Stage(beep=beep)
    stage.clip_start = beep.abs_time
    stage.start_reason = "beep"

    # look backwards for standby/ready within 30s
    for j in range(beep_idx - 1, -1, -1):
        a = anchors[j]
        if beep.abs_time - a.abs_time > 30:
            break
        if a.kind == "standby" and stage.standby is None:
            stage.standby = a
        elif a.kind == "ready" and stage.ready is None:
            stage.ready = a

    if end_idx is not None:
        end_a = anchors[end_idx]
        stage.end_command = end_a
        stage.clip_end = end_a.abs_time + (end_a.end_offset - end_a.file_offset)
        stage.end_reason = f"matched:{end_a.text}"
        stage.complete = True

        # collect gunshots between beep and end
        for j in range(beep_idx + 1, end_idx):
            if anchors[j].kind == "gunshot":
                stage.gunshots.append(anchors[j])
    else:
        # collect gunshots after beep within fallback window
        for j in range(beep_idx + 1, len(anchors)):
            a = anchors[j]
            if a.abs_time - beep.abs_time > FALLBACK_DURATION:
                break
            if a.kind == "gunshot":
                stage.gunshots.append(a)

    return stage


def _trim_fallback(
    stage: Stage,
    confirmed_intervals: list[tuple[float, float]],
    min_clip_length: float,
) -> Stage | None:
    """Trim a fallback stage so it doesn't overlap any confirmed stage.

    Returns the trimmed stage, or None if the remaining interval is too short.
    """
    orig_start = stage.clip_start
    orig_end = stage.clip_end
    stage.original_clip_start = orig_start
    stage.original_clip_end = orig_end

    remaining = _subtract_intervals(orig_start, orig_end, confirmed_intervals)

    if not remaining:
        stage.trimmed = True
        stage.trimmed_by = "fully_overlapped_by_confirmed_stages"
        return None

    # pick the longest remaining interval
    best_start, best_end = max(remaining, key=lambda iv: iv[1] - iv[0])
    best_duration = best_end - best_start

    if best_duration < min_clip_length:
        stage.trimmed = True
        stage.trimmed_by = f"remaining_interval_{best_duration:.1f}s_below_min_{min_clip_length:.1f}s"
        return None

    if best_start != orig_start or best_end != orig_end:
        stage.trimmed = True
        blockers = []
        for cs, ce in sorted(confirmed_intervals):
            if ce > orig_start and cs < orig_end:
                blockers.append(f"{cs:.2f}-{ce:.2f}")
        stage.trimmed_by = f"confirmed_stages:[{','.join(blockers)}]"
        stage.clip_start = best_start
        stage.clip_end = best_end

        if stage.start_reason.startswith("fallback"):
            stage.start_reason += "_trimmed"
        if stage.end_reason.startswith("fallback"):
            stage.end_reason += "_trimmed"

        log.info("  Fallback trimmed: %.2f-%.2f -> %.2f-%.2f (avoided %s)",
                 orig_start, orig_end, best_start, best_end, stage.trimmed_by)

    return stage


def _dedup_overlapping_stages(stages: list[Stage]) -> list[Stage]:
    """Remove stages that substantially overlap a better stage.

    Priority: confirmed > fallback. Among same type, longer phrase match wins.
    A stage is removed if >50% of its duration overlaps a higher-priority stage.
    """
    if len(stages) <= 1:
        return stages

    keep: list[Stage] = []
    for stage in stages:
        dominated = False
        for other in keep:
            overlap = _overlap_seconds(stage, other)
            shorter_dur = min(stage.duration, other.duration)
            if shorter_dur <= 0:
                continue
            if overlap / shorter_dur > 0.5:
                # one dominates the other — keep the better one
                if _stage_priority(other) >= _stage_priority(stage):
                    dominated = True
                    log.info("  Dedup: dropping stage %.2f-%.2f (%s) — overlaps %.2f-%.2f (%s)",
                             stage.clip_start, stage.clip_end, stage.start_reason,
                             other.clip_start, other.clip_end, other.start_reason)
                    break
        if not dominated:
            # check if this stage dominates any already-kept stage
            new_keep: list[Stage] = []
            for other in keep:
                overlap = _overlap_seconds(stage, other)
                shorter_dur = min(stage.duration, other.duration)
                if shorter_dur > 0 and overlap / shorter_dur > 0.5 and _stage_priority(stage) > _stage_priority(other):
                    log.info("  Dedup: replacing stage %.2f-%.2f with %.2f-%.2f",
                             other.clip_start, other.clip_end, stage.clip_start, stage.clip_end)
                else:
                    new_keep.append(other)
            new_keep.append(stage)
            keep = new_keep

    log.info("  Dedup: %d stages -> %d stages", len(stages), len(keep))
    return keep


def _overlap_seconds(a: Stage, b: Stage) -> float:
    start = max(a.clip_start, b.clip_start)
    end = min(a.clip_end, b.clip_end)
    return max(0.0, end - start)


def _stage_priority(s: Stage) -> int:
    """Higher = better. Confirmed > fallback, longer end match > shorter."""
    priority = 0
    if s.complete:
        priority += 1000
    if s.end_command:
        priority += len(s.end_command.text)
    if s.beep:
        priority += 100
    return priority


def _subtract_intervals(
    start: float,
    end: float,
    exclude: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Subtract excluded intervals from [start, end]. Return remaining pieces."""
    remaining = [(start, end)]
    for es, ee in sorted(exclude):
        new_remaining: list[tuple[float, float]] = []
        for rs, re in remaining:
            if ee <= rs or es >= re:
                new_remaining.append((rs, re))
            else:
                if rs < es:
                    new_remaining.append((rs, es))
                if re > ee:
                    new_remaining.append((ee, re))
        remaining = new_remaining
    return remaining


def _log_stage(stage: Stage, num: int, label: str = "STAGE") -> None:
    log.info("-" * 40)
    log.info("%s #%d:", label, num)

    if stage.ready:
        log.info("  ready:    file[%d] @ %.2fs '%s'", stage.ready.file_idx, stage.ready.file_offset, stage.ready.text)
    if stage.standby:
        log.info("  standby:  file[%d] @ %.2fs '%s'", stage.standby.file_idx, stage.standby.file_offset, stage.standby.text)
    if stage.beep:
        log.info("  beep:     file[%d] @ %.3fs", stage.beep.file_idx, stage.beep.file_offset)
    log.info("  gunshots: %d", len(stage.gunshots))
    if stage.end_command:
        log.info("  end:      file[%d] @ %.2fs '%s'", stage.end_command.file_idx, stage.end_command.file_offset, stage.end_command.text)
    else:
        log.warning("  end:      NOT FOUND")

    log.info("  clip:     %.2f - %.2f (%.1fs) start_reason=%s end_reason=%s complete=%s",
             stage.clip_start, stage.clip_end, stage.duration,
             stage.start_reason, stage.end_reason, stage.complete)

    if stage.trimmed:
        log.info("  trimmed:  original=%.2f-%.2f, trimmed_by=%s",
                 stage.original_clip_start or 0, stage.original_clip_end or 0, stage.trimmed_by)


# ---------------------------------------------------------------------------
# Pass 3: cut stages
# ---------------------------------------------------------------------------

def _cut_stages(
    stages: list[Stage],
    files: list[FileInfo],
    output_dir: Path,
    config: ProcessingConfig,
    debug_dir: Path,
) -> list[ManifestRow]:
    """Map each stage back to file(s) and cut."""
    rows: list[ManifestRow] = []

    for i, stage in enumerate(stages):
        log.info("=" * 60)
        log.info("CUTTING STAGE #%d", i + 1)

        if stage.complete:
            tag = ""
        elif stage.beep and not stage.end_command:
            tag = "no_end"
        elif stage.end_command and not stage.beep:
            tag = "no_start"
        else:
            tag = "incomplete"

        if not stage.complete:
            log.error(
                "Stage #%d is INCOMPLETE (%s): start_reason=%s end_reason=%s",
                i + 1, tag, stage.start_reason, stage.end_reason,
            )

        clip_start = stage.clip_start - config.start_padding
        clip_end = stage.clip_end + config.end_padding

        spans = _find_file_spans(clip_start, clip_end, files)
        if not spans:
            rows.append(ManifestRow(
                source_file="unknown",
                status="failed",
                error_message=f"Stage #{i+1}: could not map to any file",
            ))
            continue

        primary = files[spans[0][0]]
        output_name = _build_output_name(primary.path, primary.creation_str, tag)
        output_path = output_dir / output_name

        duration = clip_end - clip_start
        row = ManifestRow(
            source_file=" + ".join(files[fi].path.name for fi, _, _ in spans),
            creation_time=primary.creation_iso,
            duration=f"{duration:.3f}",
            start_offset=f"{clip_start - primary.creation_epoch:.3f}",
            end_offset=f"{clip_end - primary.creation_epoch:.3f}",
            start_reason=stage.start_reason,
            end_reason=stage.end_reason,
            confidence=f"{_stage_confidence(stage):.2f}",
        )

        _save_stage_debug(stage, i + 1, files, debug_dir)

        if output_path.exists() and not config.overwrite:
            row.status = "skipped"
            row.output_file = str(output_path)
            row.error_message = "Output already exists (use --overwrite)"
            rows.append(row)
            continue

        if config.dry_run:
            row.status = "dry_run"
            row.output_file = str(output_path)
            log.info("  DRY RUN: would write %s", output_name)
            rows.append(row)
            continue

        if duration < config.min_clip_length:
            row.status = "failed"
            row.error_message = f"Clip too short: {duration:.1f}s"
            rows.append(row)
            continue
        if duration > config.max_clip_length:
            row.status = "failed"
            row.error_message = f"Clip too long: {duration:.1f}s"
            rows.append(row)
            continue

        try:
            if len(spans) == 1:
                fi, local_start, local_end = spans[0]
                log.info("  Single-file cut: %s %.2f-%.2fs", files[fi].path.name, local_start, local_end)
                cut_clip(
                    files[fi].path, output_path, local_start, local_end,
                    accurate=config.accurate_cut,
                    min_clip_length=config.min_clip_length,
                    max_clip_length=config.max_clip_length,
                )
            else:
                source_paths = [files[fi].path for fi, _, _ in spans]
                global_start = spans[0][1]
                global_end = sum(files[fi].duration for fi, _, _ in spans[:-1]) + spans[-1][2]
                log.info(
                    "  Cross-file cut: %s, combined %.2f-%.2fs",
                    " + ".join(files[fi].path.name for fi, _, _ in spans),
                    global_start, global_end,
                )
                concat_and_cut(
                    source_paths, output_path, global_start, global_end,
                    accurate=config.accurate_cut,
                )
            row.status = "ok"
            row.output_file = str(output_path)
            log.info("  Wrote %s (%.1fs)", output_name, duration)
        except Exception as exc:
            row.status = "failed"
            row.error_message = str(exc)
            log.error("  Cut failed: %s", exc)

        rows.append(row)

    return rows


def _find_file_spans(
    abs_start: float,
    abs_end: float,
    files: list[FileInfo],
) -> list[tuple[int, float, float]]:
    """Map absolute time range to [(file_index, local_start, local_end), ...]."""
    spans: list[tuple[int, float, float]] = []
    for i, fi in enumerate(files):
        file_start = fi.creation_epoch
        file_end = fi.creation_epoch + fi.duration
        if abs_end <= file_start or abs_start >= file_end:
            continue
        local_start = max(0.0, abs_start - file_start)
        local_end = min(fi.duration, abs_end - file_start)
        spans.append((i, local_start, local_end))
    return spans


def _build_output_name(video_path: Path, creation_str: str, tag: str = "") -> str:
    stem = video_path.stem
    ts = creation_str.replace(":", "-").replace(" ", "_")
    suffix = f"__{tag}" if tag else ""
    return f"{ts}__{stem}__stage_clip{suffix}.mp4"


def _stage_confidence(stage: Stage) -> float:
    scores: list[float] = []
    if stage.beep:
        scores.append(stage.beep.score / 100.0)
    if stage.standby:
        scores.append(stage.standby.score / 100.0)
    if stage.end_command:
        scores.append(stage.end_command.score / 100.0)
    if stage.gunshots:
        scores.append(min(1.0, len(stage.gunshots) / 5.0))
    return sum(scores) / len(scores) if scores else 0.0


def _save_stage_debug(
    stage: Stage,
    stage_num: int,
    files: list[FileInfo],
    debug_dir: Path,
) -> None:
    primary_idx = stage.beep.file_idx if stage.beep else (
        stage.end_command.file_idx if stage.end_command else 0
    )

    # find beep search records from the primary file
    beep_search_debug = []
    if primary_idx < len(files):
        for rec in files[primary_idx].beep_searches:
            beep_search_debug.append(asdict(rec))

    data = {
        "stage_number": stage_num,
        "complete": stage.complete,
        "clip_start": stage.clip_start,
        "clip_end": stage.clip_end,
        "duration": stage.duration,
        "start_reason": stage.start_reason,
        "end_reason": stage.end_reason,
        "trimmed": stage.trimmed,
        "trimmed_by": stage.trimmed_by,
        "original_clip_start": stage.original_clip_start,
        "original_clip_end": stage.original_clip_end,
        "ready": asdict(stage.ready) if stage.ready else None,
        "standby": asdict(stage.standby) if stage.standby else None,
        "beep": asdict(stage.beep) if stage.beep else None,
        "beep_searches": beep_search_debug,
        "gunshots_count": len(stage.gunshots),
        "gunshot_times": [g.file_offset for g in stage.gunshots],
        "end_command": asdict(stage.end_command) if stage.end_command else None,
    }
    stem = files[primary_idx].path.stem if primary_idx < len(files) else "unknown"
    path = debug_dir / f"{stem}_stage{stage_num}_detection.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Transcription helper
# ---------------------------------------------------------------------------

def _transcribe(
    wav_path: Path,
    config: ProcessingConfig,
    cached_model: object | None,
) -> tuple[list[TranscriptSegment], object]:
    from faster_whisper import WhisperModel

    if cached_model is None:
        log.info("Loading Whisper model '%s' (device=%s, compute_type=%s) ...",
                 config.model, config.device, config.compute_type)
        cached_model = WhisperModel(
            config.model, device=config.device, compute_type=config.compute_type,
        )

    log.info("  Transcribing %s ...", wav_path.name)
    segments_iter, _info = cached_model.transcribe(
        str(wav_path),
        beam_size=5,
        word_timestamps=True,
        language="en",
        initial_prompt=(
            "USPSA practical shooting match. Range officer commands: "
            "Load and make ready. Make ready. Shooter ready. "
            "Are you ready? Stand by. Standby. "
            "If you are finished, unload and show clear. "
            "If clear, hammer down and holster. Hammer down. Holster. "
            "Range is clear. Stage is clear."
        ),
        vad_filter=True,
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

    return segments, cached_model


# ---------------------------------------------------------------------------
# Parallel worker (runs in child process)
# ---------------------------------------------------------------------------

_worker_whisper_model = None


def _worker_init(model_name: str, device: str, compute_type: str) -> None:
    """Called once per worker process to load the Whisper model."""
    global _worker_whisper_model
    from faster_whisper import WhisperModel
    _worker_whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)


def _worker_process_file(args: dict) -> dict:
    """Process a single file in a worker process. Returns serializable results."""
    global _worker_whisper_model

    video_path = Path(args["video_path"])
    wav_path = Path(args["wav_path"])
    debug_dir = Path(args["debug_dir"])
    file_idx = args["file_idx"]
    epoch = args["epoch"]
    duration = args["duration"]
    creation_str = args["creation_str"]
    creation_iso = args["creation_iso"]
    phrase_threshold = args["phrase_threshold"]
    beep_search_before = args["beep_search_before"]
    beep_search_after = args["beep_search_after"]

    fi = FileInfo(
        path=video_path,
        wav_path=wav_path,
        duration=duration,
        creation_epoch=epoch,
        creation_str=creation_str,
        creation_iso=creation_iso,
    )

    config = ProcessingConfig(
        phrase_threshold=phrase_threshold,
        beep_search_before=beep_search_before,
        beep_search_after=beep_search_after,
    )

    try:
        anchors, _ = _collect_anchors_for_file(
            fi, file_idx, config, debug_dir, _worker_whisper_model,
        )
        return {
            "file_idx": file_idx,
            "anchors": [asdict(a) for a in anchors],
            "beep_searches": [asdict(bs) for bs in fi.beep_searches],
            "error": None,
        }
    except Exception as exc:
        return {
            "file_idx": file_idx,
            "anchors": [],
            "beep_searches": [],
            "error": str(exc),
        }


def _resolve_workers(config: ProcessingConfig) -> int:
    """Return the number of worker processes to use."""
    if config.workers > 0:
        n = config.workers
    else:
        n = max(1, int((os.cpu_count() or 1) * 0.75))

    if config.device != "cpu" and n > 1:
        log.warning("GPU mode (device=%s): forcing workers=1 to avoid VRAM contention", config.device)
        n = 1

    return n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_batch(
    input_dir: Path,
    output_dir: Path,
    config: ProcessingConfig,
) -> list[ManifestRow]:
    from tqdm import tqdm

    videos = discover_videos(input_dir)
    if not videos:
        log.warning("No video files found in %s", input_dir)
        return []

    log.info("Found %d video(s) in %s", len(videos), input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = config.debug_dir or output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = debug_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    durations = {vp: get_duration(vp) for vp in videos}
    has_all_metadata = all(get_creation_time(v) is not None for v in videos)

    files: list[FileInfo] = []
    synthetic_epoch = 0.0

    for vp in videos:
        dur = durations[vp]
        creation_dt = get_creation_time(vp)

        if has_all_metadata and creation_dt is not None:
            epoch = creation_dt.timestamp()
            ts_str = creation_dt.strftime("%Y-%m-%d_%H-%M-%S")
            iso_str = creation_dt.isoformat()
        else:
            epoch = synthetic_epoch
            display_dt = get_creation_time_or_mtime(vp)
            ts_str = display_dt.strftime("%Y-%m-%d_%H-%M-%S")
            iso_str = display_dt.isoformat()

        files.append(FileInfo(
            path=vp,
            wav_path=wav_dir / f"{vp.stem}.wav",
            duration=dur,
            creation_epoch=epoch,
            creation_str=ts_str,
            creation_iso=iso_str,
        ))
        synthetic_epoch += dur

    if not has_all_metadata:
        log.info("Using synthetic timeline (files placed sequentially, total %.1fs)", synthetic_epoch)
        for i, fi in enumerate(files):
            log.info("  [%d] %s: epoch=%.1f duration=%.1fs", i, fi.path.name, fi.creation_epoch, fi.duration)

    num_workers = _resolve_workers(config)
    log.info("PASS 1: Collecting anchors from %d files (workers=%d) ...", len(files), num_workers)
    all_anchors: list[Anchor] = []

    if num_workers <= 1 or len(files) <= 1:
        # sequential mode: share one model across files
        whisper_model = None
        for i, fi in enumerate(tqdm(files, desc="Pass 1: collecting anchors")):
            try:
                anchors, whisper_model = _collect_anchors_for_file(
                    fi, i, config, debug_dir, whisper_model,
                )
                all_anchors.extend(anchors)
            except Exception as exc:
                log.error("Failed to process %s: %s", fi.path.name, exc)
    else:
        # parallel mode: each worker loads its own model
        worker_args = [
            {
                "video_path": str(fi.path),
                "wav_path": str(fi.wav_path),
                "debug_dir": str(debug_dir),
                "file_idx": i,
                "epoch": fi.creation_epoch,
                "duration": fi.duration,
                "creation_str": fi.creation_str,
                "creation_iso": fi.creation_iso,
                "phrase_threshold": config.phrase_threshold,
                "beep_search_before": config.beep_search_before,
                "beep_search_after": config.beep_search_after,
            }
            for i, fi in enumerate(files)
        ]

        log.info("Spawning %d worker processes (each loads its own Whisper model) ...", num_workers)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(config.model, config.device, config.compute_type),
        ) as executor:
            for result in tqdm(
                executor.map(_worker_process_file, worker_args),
                total=len(worker_args),
                desc="Pass 1: collecting anchors",
            ):
                if result["error"]:
                    fi = files[result["file_idx"]]
                    log.error("Failed to process %s: %s", fi.path.name, result["error"])
                else:
                    for ad in result["anchors"]:
                        all_anchors.append(Anchor(**ad))
                    fi = files[result["file_idx"]]
                    fi.beep_searches = [
                        BeepSearchRecord(**bs) for bs in result["beep_searches"]
                    ]

    log.info("PASS 1 COMPLETE: %d anchors total", len(all_anchors))

    log.info("PASS 2: Assembling stages ...")
    stages = _assemble_stages(all_anchors, min_clip_length=config.min_clip_length)
    log.info("PASS 2 COMPLETE: %d stages found", len(stages))

    if not stages:
        log.warning("No stages detected across any files")

    log.info("PASS 3: Cutting clips ...")
    rows = _cut_stages(stages, files, output_dir, config, debug_dir)

    if not config.keep_wav:
        for fi in files:
            if fi.wav_path.exists():
                fi.wav_path.unlink(missing_ok=True)

    return rows
