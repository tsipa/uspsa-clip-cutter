"""CLI entry-point using Typer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from video_stage_cutter.ffmpeg_utils import FFmpegNotFoundError, check_ffmpeg_available
from video_stage_cutter.manifest import write_manifest
from video_stage_cutter.pipeline import ProcessingConfig, run_batch

app = typer.Typer(
    name="uspsa-clip-cutter",
    help="Batch-extract USPSA stage clips from action camera videos.",
    add_completion=False,
)


@app.command()
def run(
    input_dir: Path = typer.Argument(..., help="Folder containing .mp4/.mov/.m4v source videos."),
    output_dir: Path = typer.Argument(..., help="Folder where cut clips will be saved."),
    model: str = typer.Option("small", help="Whisper model name (tiny, base, small, medium, large-v3)."),
    device: str = typer.Option("cpu", help="Device for Whisper inference (cpu or cuda)."),
    compute_type: str = typer.Option("int8", help="Compute type (int8, float16, float32)."),
    accurate_cut: bool = typer.Option(True, "--accurate-cut/--fast-cut", help="Re-encode for frame-accurate cuts, or stream-copy for speed."),
    keep_wav: bool = typer.Option(False, help="Keep extracted WAV files in the debug directory."),
    debug_dir: Path | None = typer.Option(None, help="Directory for debug/transcript files. Defaults to <output_dir>/debug."),
    start_padding: float = typer.Option(0.0, help="Seconds to include before the detected start."),
    end_padding: float = typer.Option(2.0, help="Seconds to include after the detected end command."),
    min_clip_length: float = typer.Option(5.0, help="Minimum clip duration in seconds."),
    max_clip_length: float = typer.Option(600.0, help="Maximum clip duration in seconds."),
    overwrite: bool = typer.Option(False, help="Overwrite existing output files."),
    dry_run: bool = typer.Option(False, help="Detect boundaries and write manifest/debug, but do not cut video."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging."),
) -> None:
    """Scan INPUT_DIR for videos, detect stage boundaries, and cut clips into OUTPUT_DIR."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        check_ffmpeg_available()
    except FFmpegNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if not input_dir.is_dir():
        typer.echo(f"Input directory does not exist: {input_dir}", err=True)
        raise typer.Exit(code=1)

    config = ProcessingConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        accurate_cut=accurate_cut,
        keep_wav=keep_wav,
        debug_dir=debug_dir,
        start_padding=start_padding,
        end_padding=end_padding,
        min_clip_length=min_clip_length,
        max_clip_length=max_clip_length,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    rows = run_batch(input_dir, output_dir, config)

    manifest_path = output_dir / "manifest.csv"
    write_manifest(rows, manifest_path)

    ok = sum(1 for r in rows if r.status == "ok")
    dr = sum(1 for r in rows if r.status == "dry_run")
    failed = sum(1 for r in rows if r.status == "failed")
    skipped = sum(1 for r in rows if r.status == "skipped")

    typer.echo(f"\nDone. {ok} cut, {dr} dry-run, {skipped} skipped, {failed} failed.")
    typer.echo(f"Manifest: {manifest_path}")

    if failed:
        raise typer.Exit(code=2)
