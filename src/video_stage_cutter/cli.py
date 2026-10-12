"""CLI entry-point using Typer."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from video_stage_cutter.ffmpeg_utils import (
    FFmpegFeatureError,
    FFmpegNotFoundError,
    check_ffmpeg_available,
    check_ffmpeg_features,
)
from video_stage_cutter.manifest import write_manifest
from video_stage_cutter.pipeline import ProcessingConfig, run_batch
from video_stage_cutter.power import prevent_sleep

app = typer.Typer(
    name="uspsa-clip-cutter",
    help="Batch-extract USPSA stage clips from action camera videos.",
    add_completion=False,
)


@app.command()
def run(
    input_dir: Path = typer.Argument(..., help="Folder containing .mp4/.mov/.m4v source videos."),
    output_dir: Path = typer.Argument(..., help="Folder where cut clips will be saved."),
    model: str = typer.Option("large-v3", help="Whisper model name (tiny, base, small, medium, large-v3)."),
    device: str = typer.Option("cpu", help="Device for Whisper inference (cpu or cuda)."),
    compute_type: str = typer.Option("int8", help="Compute type (int8, float16, float32)."),
    batched: bool = typer.Option(False, "--batched", "--batched-transcription", help="Transcribe audio chunks in batches using one Whisper model; may use more GPU memory."),
    batch_size: int = typer.Option(2, min=1, help="Audio chunks per inference batch; used with --batched. Start with 2 on an 8 GB GPU."),
    accurate_cut: bool = typer.Option(True, "--accurate-cut/--fast-cut", help="Re-encode for frame-accurate cuts, or stream-copy for speed."),
    compress: bool = typer.Option(False, "--compress", help="Compress to at most 1080p/30fps HEVC slow CRF 26 + AAC 96k, without upscaling. Overrides --fast-cut."),
    compress_crf: int = typer.Option(26, min=0, max=51, help="CPU HEVC quality for --compress: lower = better quality/larger files, higher = smaller files."),
    gpu_encode: bool = typer.Option(False, "--gpu-encode", help="Encode HEVC on NVIDIA NVENC (p7/hq). Implies --compress; independent of Whisper --device."),
    gpu_cq: int = typer.Option(26, min=1, max=51, help="NVENC quality for --gpu-encode: lower = better/larger. Separate from CPU --compress-crf."),
    stabilize: bool = typer.Option(False, "--stabilize", help="Apply two-pass vid.stab image stabilization. Overrides --fast-cut."),
    stabilize_smoothing: int = typer.Option(10, min=0, max=1000, help="Stabilization radius in source frames (no FPS conversion). Default: static_10 benchmark profile."),
    keep_wav: bool = typer.Option(False, help="Keep extracted WAV files in the debug directory."),
    debug_dir: Path | None = typer.Option(None, help="Directory for debug/transcript files. Defaults to <output_dir>/debug."),
    start_padding: float = typer.Option(10.0, help="Seconds to include before the detected start (beep or 'are you ready')."),
    end_padding: float = typer.Option(10.0, help="Seconds to include after the detected end command."),
    min_clip_length: float = typer.Option(5.0, help="Minimum clip duration in seconds."),
    max_clip_length: float = typer.Option(600.0, help="Maximum clip duration in seconds."),
    overwrite: bool = typer.Option(False, help="Overwrite existing output files."),
    dry_run: bool = typer.Option(False, help="Detect boundaries and write manifest/debug, but do not cut video."),
    phrase_threshold: float = typer.Option(70.0, help="Fuzzy matching threshold for phrase detection (0-100). Lower = more lenient."),
    beep_search_before: float = typer.Option(0.25, help="Seconds before standby end to start searching for the beep."),
    beep_search_after: float = typer.Option(10.0, help="Seconds after standby end to stop searching for the beep."),
    reuse_transcripts: bool = typer.Option(False, help="Reuse existing transcript JSON files from the debug directory instead of re-transcribing."),
    restore: bool = typer.Option(False, "--restore", help="Reuse completed transcripts and resume partial transcription checkpoints from the output debug directory."),
    workers: int = typer.Option(1, help="Number of parallel transcription workers. Each loads its own Whisper model (~3GB RAM)."),
    video_workers: int = typer.Option(1, min=1, help="Number of clips processed in parallel by FFmpeg (including stabilization). Independent of Whisper --workers."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging."),
) -> None:
    """Scan INPUT_DIR for videos, detect stage boundaries, and cut clips into OUTPUT_DIR."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    compress = compress or gpu_encode
    try:
        check_ffmpeg_available()
        if not dry_run:
            check_ffmpeg_features(compress=compress, stabilize=stabilize, gpu_encode=gpu_encode)
    except (FFmpegNotFoundError, FFmpegFeatureError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if not input_dir.is_dir():
        typer.echo(f"Input directory does not exist: {input_dir}", err=True)
        raise typer.Exit(code=1)

    if (compress or stabilize) and not accurate_cut:
        logging.getLogger(__name__).info("Compression/stabilization requires re-encoding; overriding --fast-cut")
        accurate_cut = True

    config = ProcessingConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        batched=batched,
        batch_size=batch_size,
        accurate_cut=accurate_cut,
        compress=compress,
        compress_crf=compress_crf,
        stabilize=stabilize,
        stabilize_smoothing=stabilize_smoothing,
        gpu_encode=gpu_encode,
        gpu_cq=gpu_cq,
        keep_wav=keep_wav,
        debug_dir=debug_dir,
        start_padding=start_padding,
        end_padding=end_padding,
        min_clip_length=min_clip_length,
        max_clip_length=max_clip_length,
        overwrite=overwrite,
        dry_run=dry_run,
        phrase_threshold=phrase_threshold,
        beep_search_before=beep_search_before,
        beep_search_after=beep_search_after,
        reuse_transcripts=reuse_transcripts,
        restore=restore,
        workers=workers,
        video_workers=video_workers,
    )

    manifest_path = output_dir / "manifest.csv"
    with prevent_sleep():
        rows = run_batch(input_dir, output_dir, config)
        write_manifest(rows, manifest_path)

    ok = sum(1 for r in rows if r.status == "ok")
    dr = sum(1 for r in rows if r.status == "dry_run")
    failed = sum(1 for r in rows if r.status == "failed")
    skipped = sum(1 for r in rows if r.status == "skipped")

    typer.echo(f"\nDone. {ok} cut, {dr} dry-run, {skipped} skipped, {failed} failed.")
    typer.echo(f"Manifest: {manifest_path}")

    if failed:
        raise typer.Exit(code=2)
