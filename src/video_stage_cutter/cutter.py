"""High-level clip cutting logic with boundary validation."""

from __future__ import annotations

import logging
from pathlib import Path

from video_stage_cutter.ffmpeg_utils import cut_video

log = logging.getLogger(__name__)


def cut_clip(
    video_path: Path,
    output_path: Path,
    start: float,
    end: float,
    accurate: bool = True,
    min_clip_length: float = 5.0,
    max_clip_length: float = 600.0,
) -> Path:
    """Validate boundaries and cut a clip.

    Raises ``ValueError`` if the resulting clip would be outside the
    allowed length range.
    """
    if start < 0:
        log.warning("start (%.3f) is negative, clamping to 0", start)
        start = 0.0

    duration = end - start
    if duration < min_clip_length:
        raise ValueError(
            f"Clip would be {duration:.1f}s, below minimum {min_clip_length:.1f}s"
        )
    if duration > max_clip_length:
        raise ValueError(
            f"Clip would be {duration:.1f}s, above maximum {max_clip_length:.1f}s"
        )

    log.info(
        "Cutting clip: %.2f–%.2f s (%.1f s) → %s [%s]",
        start, end, duration, output_path.name, "accurate" if accurate else "fast",
    )
    return cut_video(video_path, output_path, start, end, accurate=accurate)
