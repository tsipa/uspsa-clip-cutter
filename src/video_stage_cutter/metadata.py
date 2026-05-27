"""Read creation-time metadata from video files via ffprobe."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# GoPro Hero5+: GX<chapter><clip_id> or GH<chapter><clip_id>
_GOPRO_NEW = re.compile(r"^G[XH](\d{2})(\d{4})$", re.IGNORECASE)
# GoPro older: GOPR<clip_id> (first chapter), GP<chapter><clip_id> (cont)
_GOPRO_FIRST = re.compile(r"^GOPR(\d{4})$", re.IGNORECASE)
_GOPRO_CONT = re.compile(r"^GP(\d{2})(\d{4})$", re.IGNORECASE)


def get_creation_time(video_path: Path) -> datetime | None:
    """Return a timezone-aware creation timestamp, or None if unavailable.

    Only returns a real timestamp from ffprobe metadata.
    Does NOT fall back to mtime — caller decides what to do.
    """
    try:
        probe = _ffprobe_json(video_path)
        ts = _extract_creation_time(probe)
        if ts is not None:
            return ts
    except Exception:
        log.debug("ffprobe metadata read failed for %s", video_path)

    return None


def get_creation_time_or_mtime(video_path: Path) -> datetime:
    """Return creation timestamp, falling back to filesystem mtime."""
    ts = get_creation_time(video_path)
    if ts is not None:
        return ts
    mtime = os.path.getmtime(video_path)
    log.debug("No creation_time for %s, using mtime", video_path.name)
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


def filename_sort_key(path: Path) -> tuple:
    """Sort key that handles GoPro naming correctly.

    GoPro: sort by (clip_id, chapter) so chapters stay together.
    Everything else: lexicographic by lowercased stem.
    """
    stem = path.stem

    m = _GOPRO_NEW.match(stem)
    if m:
        chapter, clip_id = int(m.group(1)), int(m.group(2))
        return (0, clip_id, chapter, "")

    m = _GOPRO_FIRST.match(stem)
    if m:
        clip_id = int(m.group(1))
        return (0, clip_id, 0, "")

    m = _GOPRO_CONT.match(stem)
    if m:
        chapter, clip_id = int(m.group(1)), int(m.group(2))
        return (0, clip_id, chapter, "")

    return (1, 0, 0, stem.lower())


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
