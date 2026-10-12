"""Wrappers around ffmpeg / ffprobe CLI commands."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
from fractions import Fraction
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
        requirements.append(("-encoders", {"libx265", "aac"}))
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
                "Install a full ffmpeg build with libx265, AAC and libvidstab."
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
    start: float = 0.0,
    source_fps: float = 30.0,
) -> Path:
    """Render with optional constant-quality HEVC and vid.stab stabilization.

    A relative transform filename in a private working directory avoids
    Windows drive-letter escaping in filter strings.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Clip duration must be positive and finite")
    if not math.isfinite(start) or start < 0:
        raise ValueError("Clip start must be nonnegative and finite")
    if not 0 <= compress_crf <= 51:
        raise ValueError("Compression CRF must be between 0 and 51")
    if not 0 <= stabilize_smoothing <= 1000:
        raise ValueError("Stabilization smoothing must be between 0 and 1000 frames")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if compress:
        if not stabilize:
            filters.append("fps=30")
        filters += [
            "scale=w='min(iw,1920)':h='min(ih,1080)':"
            "force_original_aspect_ratio=decrease:force_divisible_by=2",
        ]
        if stabilize:
            # Prevent encoder pixel-format negotiation from making the two
            # stabilization passes see different scaled pixel data.
            filters.append("format=yuv420p")
        log.info(
            "Compression: up to 1920x1080 / 30 fps, HEVC slow CRF %d, "
            "AAC 96 kbit/s; constant quality, no file size limit (%.1fs)",
            compress_crf, duration,
        )

    with tempfile.TemporaryDirectory(prefix="clip_render_") as tmp:
        work_dir = Path(tmp)
        # Input -ss seeks to a nearby keyframe; default accurate seeking discards
        # preroll before either filter pass. Input -t bounds demuxing after that
        # seek, and output -t caps encoded duration. Keep timestamps on the same
        # origin for audio and video instead of resetting each stream separately.
        base = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            *input_args, "-t", f"{duration:.3f}",
        ]
        if stabilize:
            if not math.isfinite(source_fps) or source_fps <= 0:
                raise ValueError("Source frame rate must be positive and finite")
            # The CLI radius is expressed in frames at 30 fps, so 30 means
            # approximately one second on either side at any source frame rate.
            smoothing = int(stabilize_smoothing * source_fps / 30 + 0.5)
            smoothing = max(1 if stabilize_smoothing else 0, smoothing)
            if smoothing > 1000:
                log.warning("Source-rate smoothing exceeds vid.stab's limit; clamping to 1000 frames")
                smoothing = 1000
            log.info("Stabilization: vid.stab at %.3f fps, smoothing=%d source frames", source_fps, smoothing)
            detection = [*filters, "vidstabdetect=shakiness=5:accuracy=15:result=transforms.trf"]
            _run_render([
                *base, "-map", "0:v:0", "-vf", ",".join(detection),
                "-an", "-f", "null", "-",
            ], work_dir)
            filters.append(
                "vidstabtransform=input=transforms.trf:"
                f"smoothing={smoothing}:optzoom=0:zoom=5:crop=black:interpol=bicubic"
            )
            if compress:
                filters.append("fps=30")
        video_args = ["-vf", ",".join(filters)] if filters else []
        if compress:
            video_args += [
                "-c:v", "libx265", "-preset", "slow", "-crf", str(compress_crf),
                "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
            ]
            audio_args = [
                "-c:a", "aac", "-b:a", "96k",
                "-ar", "48000", "-ac", "2",
            ]
        else:
            video_args += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"]
            audio_args = ["-c:a", "aac", "-b:a", "192k"]
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
            ["-i", str(video_path.resolve())],
            output_path, end - start, compress=compress, compress_crf=compress_crf,
            stabilize=stabilize, stabilize_smoothing=stabilize_smoothing,
            start=start,
            source_fps=get_frame_rate(video_path) if stabilize else 30.0,
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


def get_frame_rate(video_path: Path) -> float:
    """Read the first video stream's average FPS, with nominal FPS as fallback."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate", "-of", "json",
        str(video_path),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr}")
    streams = json.loads(result.stdout).get("streams", [])
    if streams:
        for name in ("avg_frame_rate", "r_frame_rate"):
            try:
                fps = float(Fraction(streams[0].get(name, "0/0")))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            if math.isfinite(fps) and fps > 0:
                return fps
    raise RuntimeError(f"Cannot determine video frame rate for {video_path}")


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

    # Explicit durations let the concat demuxer seek straight to the relevant
    # file/keyframe instead of decoding everything before the clip. Probing
    # first also avoids leaving a temporary list behind if a probe fails.
    durations = [get_duration(vp) for vp in video_paths] if compress or stabilize else []
    source_fps = 30.0
    if stabilize:
        # Use the file containing the clip start, including seeks past whole
        # files. Compatible camera chapters normally have a constant frame rate.
        offset = start
        for vp, file_duration in zip(video_paths, durations):
            if offset < file_duration:
                source_fps = get_frame_rate(vp)
                break
            offset -= file_duration
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="concat_", encoding="utf-8",
    ) as f:
        for i, vp in enumerate(video_paths):
            escaped = vp.resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
            if durations:
                f.write(f"duration {durations[i]:.6f}\n")
        concat_list = Path(f.name)

    try:
        if compress or stabilize:
            return _render_processed_clip(
                ["-f", "concat", "-safe", "0", "-i", str(concat_list.resolve())],
                output_path, end - start, compress=compress, compress_crf=compress_crf,
                stabilize=stabilize, stabilize_smoothing=stabilize_smoothing,
                start=start,
                source_fps=source_fps,
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
