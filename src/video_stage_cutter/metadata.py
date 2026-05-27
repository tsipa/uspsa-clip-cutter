"""Read creation-time metadata from video files via ffprobe."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def get_creation_time(video_path: Path) -> datetime:
    """Return a timezone-aware creation timestamp for *video_path*.

    Strategy:
    1. Try ffprobe ``format.tags.creation_time``.
    2. Try stream-level ``tags.creation_time``.
    3. Fall back to filesystem modification time.
    """
    try:
        probe = _ffprobe_json(video_path)
        ts = _extract_creation_time(probe)
        if ts is not None:
            return ts
    except Exception:
        log.debug("ffprobe metadata read failed for %s, using mtime", video_path)

    mtime = os.path.getmtime(video_path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


def _ffprobe_json(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def _extract_creation_time(probe: dict) -> datetime | None:
    raw = None
    fmt_tags = probe.get("format", {}).get("tags", {})
    raw = fmt_tags.get("creation_time") or fmt_tags.get("com.apple.quicktime.creationdate")

    if raw is None:
        for stream in probe.get("streams", []):
            stags = stream.get("tags", {})
            raw = stags.get("creation_time")
            if raw:
                break

    if raw is None:
        return None

    for pattern in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(raw, pattern)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    log.warning("Could not parse creation_time '%s'", raw)
    return None
