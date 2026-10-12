"""Encoding contracts and real ffmpeg checks for compressed/stabilized clips."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from video_stage_cutter import cli, ffmpeg_utils, pipeline
from video_stage_cutter.ffmpeg_utils import concat_and_cut, cut_video


@pytest.mark.parametrize("duration", [60, 600])
def test_compression_quality_does_not_depend_on_duration(tmp_path, monkeypatch, duration):
    render = Mock()
    monkeypatch.setattr(ffmpeg_utils, "_run_render", render)
    cut_video(tmp_path / "input.mp4", tmp_path / "out.mp4", 5, 5 + duration,
              accurate=False, compress=True)
    # One CRF encode for any length; no size target or second encode pass.
    render.assert_called_once()
    cmd = render.call_args.args[0]
    assert cmd[cmd.index("-crf") + 1] == "26"
    assert cmd[cmd.index("-preset") + 1] == "slow"
    assert cmd[cmd.index("-c:v") + 1] == "libx265"
    assert cmd[cmd.index("-c:a") + 1] == "libopus"
    assert cmd[cmd.index("-compression_level") + 1] == "10"
    assert cmd[cmd.index("-vbr") + 1] == "on"
    assert cmd[cmd.index("-t") + 1] == f"{duration:.3f}"
    assert not {"-fs", "-b:v", "-pass", "-x265-params"}.intersection(cmd)


def test_stabilization_analysis_matches_concat_encode(tmp_path, monkeypatch):
    render = Mock()
    monkeypatch.setattr(ffmpeg_utils, "_run_render", render)
    concat_and_cut([tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path / "out.mp4",
                   55, 70, accurate=False, compress=True, stabilize=True,
                   stabilize_smoothing=15)
    assert render.call_count == 2
    detection, encode = [call.args[0] for call in render.call_args_list]
    detection_filters = detection[detection.index("-vf") + 1]
    encode_filters = encode[encode.index("-vf") + 1]
    assert detection_filters.split(",vidstabdetect")[0] == encode_filters.split(",vidstabtransform")[0]
    assert detection_filters.startswith("trim=start=55.000:duration=15.000,setpts=PTS-STARTPTS,")
    assert "smoothing=15" in encode_filters
    assert "atrim=start=55.000:duration=15.000,asetpts=PTS-STARTPTS" in encode
    assert render.call_args_list[0].args[1] == render.call_args_list[1].args[1]
    assert not render.call_args.args[1].exists()  # private transforms cleaned up
    concat_path = Path(detection[detection.index("-i") + 1])
    assert not concat_path.exists()


def test_cli_passes_profile_and_overrides_fast_cut(tmp_path, monkeypatch):
    check = Mock()
    batch = Mock(return_value=[])
    monkeypatch.setattr(cli, "check_ffmpeg_available", Mock())
    monkeypatch.setattr(cli, "check_ffmpeg_features", check)
    monkeypatch.setattr(cli, "run_batch", batch)
    monkeypatch.setattr(cli, "write_manifest", Mock())
    result = CliRunner().invoke(cli.app, [
        str(tmp_path), str(tmp_path / "out"), "--compress", "--compress-crf", "28",
        "--stabilize", "--stabilize-smoothing", "15", "--fast-cut",
    ])
    assert result.exit_code == 0, result.output
    config = batch.call_args.args[2]
    assert config.compress and config.stabilize and config.accurate_cut
    assert config.compress_crf == 28 and config.stabilize_smoothing == 15
    check.assert_called_once_with(compress=True, stabilize=True)


@pytest.mark.parametrize("option,value", [("--compress-crf", "52"), ("--stabilize-smoothing", "-1")])
def test_cli_rejects_invalid_profile_values(tmp_path, option, value):
    result = CliRunner().invoke(cli.app, [str(tmp_path), str(tmp_path / "out"), option, value])
    assert result.exit_code == 2


def test_cli_dry_run_skips_feature_checks(tmp_path, monkeypatch):
    check = Mock()
    monkeypatch.setattr(cli, "check_ffmpeg_available", Mock())
    monkeypatch.setattr(cli, "check_ffmpeg_features", check)
    monkeypatch.setattr(cli, "run_batch", Mock(return_value=[]))
    monkeypatch.setattr(cli, "write_manifest", Mock())
    result = CliRunner().invoke(cli.app, [str(tmp_path), str(tmp_path / "out"),
                                         "--compress", "--stabilize", "--dry-run"])
    assert result.exit_code == 0, result.output
    check.assert_not_called()


def test_missing_features_fail_with_actionable_error(monkeypatch):
    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", Mock(return_value=
                        subprocess.CompletedProcess([], 0, stdout=" V..... libx264 H.264\n")))
    with pytest.raises(ffmpeg_utils.FFmpegFeatureError, match="libx265"):
        ffmpeg_utils.check_ffmpeg_features(compress=True)


@pytest.mark.parametrize("cross_file", [False, True])
def test_pipeline_passes_output_options(tmp_path, monkeypatch, cross_file):
    files = [pipeline.FileInfo(
        path=tmp_path / f"source{i}.mp4", wav_path=tmp_path / f"source{i}.wav",
        duration=60, creation_epoch=1000 + 60 * i,
        creation_str=f"2026-10-11_12-00-0{i}", creation_iso="2026-10-11T12:00:00",
    ) for i in range(2)]
    stage = pipeline.Stage(clip_start=1050, clip_end=1070 if cross_file else 1058, complete=True)
    cut = Mock()
    monkeypatch.setattr(pipeline, "concat_and_cut" if cross_file else "cut_clip", cut)
    monkeypatch.setattr(pipeline, "_save_stage_debug", Mock())
    config = pipeline.ProcessingConfig(start_padding=0, end_padding=0, compress=True,
                                       compress_crf=28, stabilize=True, stabilize_smoothing=15)
    rows = pipeline._cut_stages([stage], files, tmp_path / "out", config, tmp_path)
    assert rows[0].status == "ok"
    assert cut.call_args.kwargs["compress"] and cut.call_args.kwargs["stabilize"]
    assert cut.call_args.kwargs["compress_crf"] == 28
    assert cut.call_args.kwargs["stabilize_smoothing"] == 15


@pytest.fixture(scope="module")
def ffmpeg_sources(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe unavailable")
    try:
        ffmpeg_utils.check_ffmpeg_features(compress=True, stabilize=True)
    except ffmpeg_utils.FFmpegFeatureError as exc:
        pytest.skip(str(exc))
    folder = tmp_path_factory.mktemp("video output")
    paths = [folder / "camera's first.mp4", folder / "second video.mp4"]
    for path in paths:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=60",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "1.5", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ], check=True, capture_output=True, text=True)
    return paths


@pytest.mark.ffmpeg
@pytest.mark.parametrize("mode", ["compressed", "combined", "stabilized", "silent_portrait"])
def test_real_ffmpeg_output(ffmpeg_sources, tmp_path, mode):
    output = tmp_path / "clip.mp4"
    if mode == "combined":
        concat_and_cut(ffmpeg_sources, output, 1, 2, accurate=False,
                       compress=True, stabilize=True, stabilize_smoothing=5)
    elif mode == "silent_portrait":
        source = tmp_path / "portrait.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=60",
            "-t", "1.5", "-c:v", "libx264", "-preset", "ultrafast", str(source),
        ], check=True, capture_output=True, text=True)
        cut_video(source, output, 0.25, 1.25, compress=True)
    else:
        cut_video(ffmpeg_sources[0], output, 0.25, 1.25, accurate=False,
                  compress=mode == "compressed", stabilize=mode == "stabilized",
                  stabilize_smoothing=5)
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output),
    ], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert float(data["format"]["duration"]) == pytest.approx(1.0, abs=0.1)
    if mode == "stabilized":
        assert video["codec_name"] == "h264"
        assert (video["width"], video["height"]) == (320, 180)
        assert video["avg_frame_rate"] == "60/1"
    else:
        assert video["codec_name"] == "hevc"
        assert video["codec_tag_string"] == "hvc1"
        assert (video["width"], video["height"]) == (1920, 1080)
        assert video["avg_frame_rate"] == "30/1"
        assert video["pix_fmt"] == "yuv420p"
        assert video["sample_aspect_ratio"] == "1:1"
    audio = [s for s in data["streams"] if s["codec_type"] == "audio"]
    if mode == "silent_portrait":
        assert not audio
    else:
        assert audio[0]["codec_name"] == ("aac" if mode == "stabilized" else "opus")
    # Check both streams can be decoded all the way through.
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
                   check=True, capture_output=True, text=True)
