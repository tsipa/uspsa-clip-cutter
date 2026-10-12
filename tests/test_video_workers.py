"""Parallel clip scheduling, ordered results and shared-output safety."""

from threading import Barrier, Event, Lock, get_ident
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from video_stage_cutter import cli, pipeline


@pytest.fixture
def batch(tmp_path):
    files = [pipeline.FileInfo(
        path=tmp_path / f"source{i}.mp4", wav_path=tmp_path / f"source{i}.wav",
        duration=60, creation_epoch=60 * i,
        creation_str=f"2026-10-11_12-00-0{i}", creation_iso="2026-10-11T12:00:00",
    ) for i in range(4)]
    stages = [pipeline.Stage(clip_start=60 * i + 5, clip_end=60 * i + 15,
                            complete=True) for i in range(4)]
    output = tmp_path / "out"
    debug = output / "debug"
    debug.mkdir(parents=True)
    config = pipeline.ProcessingConfig(start_padding=0, end_padding=0, video_workers=2)
    return files, stages, output, debug, config


def run(batch):
    files, stages, output, debug, config = batch
    return pipeline._cut_stages(stages, files, output, config, debug)


def test_two_workers_overlap_and_respect_limit(batch, monkeypatch):
    barrier, lock = Barrier(2), Lock()
    active = maximum = 0

    def cut(source, output, *args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            barrier.wait(timeout=5)
            output.touch()
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(pipeline, "cut_clip", cut)
    rows = run(batch)
    assert maximum == 2
    assert [row.status for row in rows] == ["ok"] * 4
    assert [row.source_file for row in rows] == [f"source{i}.mp4" for i in range(4)]
    assert len(list(batch[3].glob("*_detection.json"))) == 4


def test_completion_order_does_not_change_manifest_order(batch, monkeypatch):
    finished_second = Event()
    completion = []

    def cut(source, *args, **kwargs):
        if source.stem == "source0":
            assert finished_second.wait(timeout=5)
        completion.append(source.stem)
        if source.stem == "source1":
            finished_second.set()

    monkeypatch.setattr(pipeline, "cut_clip", cut)
    rows = run(batch)
    assert completion.index("source1") < completion.index("source0")
    assert [row.source_file for row in rows] == [f"source{i}.mp4" for i in range(4)]
    assert all(row.status == "ok" for row in rows)


def test_worker_failure_does_not_stop_other_clips(batch, monkeypatch):
    def cut(source, *args, **kwargs):
        if source.stem == "source1":
            raise RuntimeError("encoder failed")

    monkeypatch.setattr(pipeline, "cut_clip", cut)
    rows = run(batch)
    assert [row.status for row in rows] == ["ok", "failed", "ok", "ok"]
    assert rows[1].error_message == "encoder failed"


def test_default_worker_runs_sequentially_in_caller_thread(batch, monkeypatch):
    batch[4].video_workers = pipeline.ProcessingConfig().video_workers
    seen = []
    caller = get_ident()
    monkeypatch.setattr(pipeline, "cut_clip", lambda source, *a, **kw:
                        seen.append((source.stem, get_ident())))
    rows = run(batch)
    assert seen == [(f"source{i}", caller) for i in range(4)]
    assert all(row.status == "ok" for row in rows)


@pytest.mark.parametrize("overwrite", [False, True])
def test_same_output_is_serialized_and_preserves_overwrite(batch, monkeypatch, overwrite):
    batch[1][:] = [batch[1][0], batch[1][0]]
    batch[4].overwrite = overwrite
    calls = []

    def cut(source, output, *args, **kwargs):
        if calls:
            assert output.read_text() == "first finished"
        calls.append(output)
        output.write_text("first finished")

    monkeypatch.setattr(pipeline, "cut_clip", cut)
    rows = run(batch)
    assert [row.status for row in rows] == ["ok", "ok" if overwrite else "skipped"]
    assert len(calls) == (2 if overwrite else 1)


@pytest.mark.parametrize("mode", ["existing", "dry_run", "too_short", "too_long", "unmapped"])
def test_non_rendering_stages_are_handled_without_workers(batch, monkeypatch, mode):
    files, stages, output, debug, config = batch
    cut = Mock()
    monkeypatch.setattr(pipeline, "cut_clip", cut)
    if mode == "existing":
        for file in files:
            (output / pipeline._build_output_name(file.path, file.creation_str)).touch()
        expected = "skipped"
    elif mode == "dry_run":
        config.dry_run = True
        expected = "dry_run"
    elif mode == "too_short":
        config.min_clip_length = 20
        expected = "failed"
    elif mode == "too_long":
        config.max_clip_length = 5
        expected = "failed"
    else:
        stages[:] = [pipeline.Stage(clip_start=1000, clip_end=1010, complete=True)]
        expected = "failed"
    assert all(row.status == expected for row in run(batch))
    cut.assert_not_called()


def test_invalid_worker_count_rejected_by_pipeline(batch):
    batch[4].video_workers = 0
    with pytest.raises(ValueError, match="Video workers must be at least 1"):
        run(batch)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_invalid_video_workers(tmp_path, value):
    result = CliRunner().invoke(cli.app, [str(tmp_path), str(tmp_path / "out"),
                                         "--video-workers", value])
    assert result.exit_code == 2


def test_cli_passes_video_workers_independently_of_whisper(tmp_path, monkeypatch):
    batch = Mock(return_value=[])
    monkeypatch.setattr(cli, "check_ffmpeg_available", Mock())
    monkeypatch.setattr(cli, "run_batch", batch)
    monkeypatch.setattr(cli, "write_manifest", Mock())
    result = CliRunner().invoke(cli.app, [str(tmp_path), str(tmp_path / "out"),
                                         "--video-workers", "2", "--dry-run"])
    assert result.exit_code == 0, result.output
    config = batch.call_args.args[2]
    assert config.video_workers == 2 and config.workers == 1
