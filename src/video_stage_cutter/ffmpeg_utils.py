"""Wrappers around ffmpeg / ffprobe CLI commands."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg or ffprobe is not on PATH."""


def check_ffmpeg_available() -> None:
    """Verify that both ``ffmpeg`` and ``ffprobe`` are reachable."""
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise FFmpegNotFoundError(
                f"'{name}' was not found on PATH. "
                f"Install ffmpeg (e.g. `winget install Gyan.FFmpeg` on Windows) "
                f"and restart your terminal."
            )


def extract_audio(video_path: Path, wav_path: Path) -> Path:
    """Extract mono 16 kHz WAV from *video_path*."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path),
    ]
    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed for {video_path}:\n{result.stderr}"
        )
    return wav_path


def cut_video(
    video_path: Path,
    output_path: Path,
    start: float,
    end: float,
    accurate: bool = True,
) -> Path:
    """Cut a segment from *video_path* and write to *output_path*.

    When *accurate* is True the segment is re-encoded for frame-accurate cuts.
    When False, stream-copy is used (fast but may start on a prior keyframe).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if accurate:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", str(video_path),
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", str(video_path),
            "-c", "copy",
            str(output_path),
        ]

    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg cut failed for {video_path}:\n{result.stderr}"
        )
    return output_path


def get_duration(video_path: Path) -> float:
    """Return duration of *video_path* in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr}")
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def concat_and_cut(
    video_paths: list[Path],
    output_path: Path,
    start: float,
    end: float,
    accurate: bool = True,
) -> Path:
    """Concatenate *video_paths* and cut from *start* to *end* on the combined timeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="concat_",
    ) as f:
        for vp in video_paths:
            f.write(f"file '{vp}'\n")
        concat_list = Path(f.name)

    try:
        if accurate:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-c", "copy",
                str(output_path),
            ]

        log.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat+cut failed:\n{result.stderr}")
    finally:
        concat_list.unlink(missing_ok=True)

    return output_path
