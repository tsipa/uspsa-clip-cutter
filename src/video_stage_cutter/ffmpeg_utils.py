"""Wrappers around ffmpeg / ffprobe CLI commands."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg or ffprobe is not on PATH."""


class FFmpegFeatureError(RuntimeError):
    """Raised when the installed ffmpeg lacks a requested encoder or filter."""


def check_ffmpeg_features(*, compress: bool = False, stabilize: bool = False) -> None:
    """Fail before transcription if optional rendering features are unavailable."""
    requirements = []
    if compress:
        requirements.append(("-encoders", {"libx265", "libopus"}))
    if stabilize:
        requirements.append(("-filters", {"vidstabdetect", "vidstabtransform"}))
    for flag, required in requirements:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", flag], capture_output=True, text=True,
        )
        available = {
            fields[1] for line in result.stdout.splitlines()
            if len(fields := line.split()) >= 2
        }
        missing = required - available
        if result.returncode != 0 or missing:
            raise FFmpegFeatureError(
                "Installed ffmpeg does not support the requested output profile. "
                f"Required features: {', '.join(sorted(required))}. "
                "Install a full ffmpeg build with libx265, libopus and libvidstab."
            )


def _run_render(cmd: list[str], cwd: Path) -> None:
    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg rendering failed:\n{result.stderr}")


def _render_processed_clip(
    input_args: list[str],
    output_path: Path,
    duration: float,
    *,
    compress: bool,
    compress_crf: int,
    stabilize: bool,
    stabilize_smoothing: int,
    trim_start: float | None = None,
) -> Path:
    """Render with optional constant-quality HEVC and vid.stab stabilization.

    A relative transform filename in a private working directory avoids
    Windows drive-letter escaping in filter strings.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Clip duration must be positive and finite")
    if not 0 <= compress_crf <= 51:
        raise ValueError("Compression CRF must be between 0 and 51")
    if not 0 <= stabilize_smoothing <= 1000:
        raise ValueError("Stabilization smoothing must be between 0 and 1000 frames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    # The concat demuxer cannot reliably seek. Trim before motion analysis so
    # analysis and encoding see exactly the same frames on the combined timeline.
    if trim_start is not None:
        filters += [f"trim=start={trim_start:.3f}:duration={duration:.3f}", "setpts=PTS-STARTPTS"]
    if compress:
        filters += [
            "fps=30",
            "scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2",
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",
        ]
        log.info(
            "Compression: 1920x1080 / 30 fps, HEVC slow CRF %d, "
            "Opus 64 kbit/s; constant quality, no file size limit (%.1fs)",
            compress_crf, duration,
        )

    with tempfile.TemporaryDirectory(prefix="clip_render_") as tmp:
        work_dir = Path(tmp)
        base = ["ffmpeg", "-y", *input_args, "-t", f"{duration:.3f}"]
        if stabilize:
            log.info("Stabilization: vid.stab, smoothing=%d frames", stabilize_smoothing)
            detection = [*filters, "vidstabdetect=shakiness=5:accuracy=15:result=transforms.trf"]
            _run_render([
                *base, "-map", "0:v:0", "-vf", ",".join(detection),
                "-an", "-f", "null", "-",
            ], work_dir)
            filters.append(
                "vidstabtransform=input=transforms.trf:"
                f"smoothing={stabilize_smoothing}:optzoom=1:interpol=bicubic"
            )
        video_args = ["-vf", ",".join(filters)] if filters else []
        if compress:
            video_args += [
                "-c:v", "libx265", "-preset", "slow", "-crf", str(compress_crf),
                "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
            ]
            audio_args = [
                "-c:a", "libopus", "-b:a", "64k", "-vbr", "on",
                "-compression_level", "10", "-application", "audio",
                "-ar", "48000", "-ac", "2",
            ]
        else:
            video_args += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]
            audio_args = ["-c:a", "aac", "-b:a", "192k"]
        if trim_start is not None:
            audio_args += [
                "-af", f"atrim=start={trim_start:.3f}:duration={duration:.3f},asetpts=PTS-STARTPTS",
            ]
        _run_render([
            *base, "-map", "0:v:0", "-map", "0:a:0?", *video_args, *audio_args,
            "-movflags", "+faststart", str(output_path.resolve()),
        ], work_dir)
    return output_path


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
    *,
    compress: bool = False,
    compress_crf: int = 26,
    stabilize: bool = False,
    stabilize_smoothing: int = 30,
) -> Path:
    """Cut a segment from *video_path* and write to *output_path*.

    When *accurate* is True the segment is re-encoded for frame-accurate cuts.
    When False, stream-copy is used (fast but may start on a prior keyframe).
    Compression or stabilization always re-encodes, overriding *accurate*.
    """
    if compress or stabilize:
        return _render_processed_clip(
            ["-ss", f"{start:.3f}", "-i", str(video_path.resolve())],
            output_path, end - start, compress=compress, compress_crf=compress_crf,
            stabilize=stabilize, stabilize_smoothing=stabilize_smoothing,
        )
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
    *,
    compress: bool = False,
    compress_crf: int = 26,
    stabilize: bool = False,
    stabilize_smoothing: int = 30,
) -> Path:
    """Concatenate *video_paths* and cut from *start* to *end* on the combined timeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="concat_", encoding="utf-8",
    ) as f:
        for vp in video_paths:
            escaped = vp.resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        concat_list = Path(f.name)

    try:
        if compress or stabilize:
            return _render_processed_clip(
                ["-f", "concat", "-safe", "0", "-i", str(concat_list.resolve())],
                output_path, end - start, compress=compress, compress_crf=compress_crf,
                stabilize=stabilize, stabilize_smoothing=stabilize_smoothing,
                trim_start=start,
            )
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
