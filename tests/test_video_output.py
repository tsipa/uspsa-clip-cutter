"""Encoding contracts and real ffmpeg checks for compressed/stabilized clips."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import pytest
import numpy as np
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
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[cmd.index("-b:a") + 1] == "96k"
    assert cmd[cmd.index("-t") + 1] == f"{duration:.3f}"
    assert not {"-fs", "-b:v", "-pass", "-x265-params"}.intersection(cmd)


def test_stabilization_analysis_matches_concat_encode(tmp_path, monkeypatch):
    render = Mock()
    monkeypatch.setattr(ffmpeg_utils, "_run_render", render)
    monkeypatch.setattr(ffmpeg_utils, "get_duration", Mock(return_value=60.0))
    concat_and_cut([tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path / "out.mp4",
                   55, 70, accurate=False, compress=True, stabilize=True,
                   stabilize_smoothing=15)
    assert render.call_count == 2
    detection, encode = [call.args[0] for call in render.call_args_list]
    detection_filters = detection[detection.index("-vf") + 1]
    encode_filters = encode[encode.index("-vf") + 1]
    assert detection_filters.split(",vidstabdetect")[0] == encode_filters.split(",vidstabtransform")[0]
    assert "smoothing=15" in encode_filters
    assert "optzoom=0:zoom=5:crop=black" in encode_filters
    assert detection[:detection.index("-map")] == encode[:encode.index("-map")]
    assert detection.index("-ss") < detection.index("-i")
    assert detection[detection.index("-ss") + 1] == "55.000"
    input_limit, output_limit = [i for i, value in enumerate(detection) if value == "-t"]
    assert input_limit < detection.index("-i") < output_limit
    assert detection[input_limit + 1] == detection[output_limit + 1] == "15.000"
    assert render.call_args_list[0].args[1] == render.call_args_list[1].args[1]
    assert not render.call_args.args[1].exists()  # private transforms cleaned up
    concat_path = Path(detection[detection.index("-i") + 1])
    assert not concat_path.exists()


def test_concat_list_declares_all_durations(tmp_path, monkeypatch):
    monkeypatch.setattr(ffmpeg_utils, "get_duration", Mock(side_effect=[120.0, 90.0]))
    contents = []

    def capture(cmd, cwd):
        contents.append(Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8"))

    monkeypatch.setattr(ffmpeg_utils, "_run_render", capture)
    concat_and_cut([tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path / "out.mp4",
                   115, 125, compress=True)
    assert "duration 120.000000\n" in contents[0]
    assert "duration 90.000000\n" in contents[0]


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
    for path, color, pulses in zip(paths, ["red", "blue"], [(2.5, 3.5), (1.5, 2.5)]):
        intervals = "+".join(f"between(t,{t},{t + 0.1})" for t in pulses)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color={color}:size=320x180:rate=60,"
            f"drawbox=color=white:t=fill:enable='{intervals}'",
            "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*880*t)*({intervals})':s=48000",
            "-t", "4", "-c:v", "libx264", "-preset", "ultrafast", "-bf", "2", "-g", "60",
            "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ], check=True, capture_output=True, text=True)
    return paths


@pytest.mark.ffmpeg
@pytest.mark.parametrize("mode", ["compressed", "combined", "stabilized", "silent_portrait"])
def test_real_ffmpeg_output(ffmpeg_sources, tmp_path, mode):
    output = tmp_path / "clip.mp4"
    if mode == "combined":
        concat_and_cut(ffmpeg_sources, output, 3.5, 4.5, accurate=False,
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
        assert (video["width"], video["height"]) == ((180, 320) if mode == "silent_portrait" else (320, 180))
        assert video["avg_frame_rate"] == "30/1"
        assert video["pix_fmt"] == "yuv420p"
        assert video["sample_aspect_ratio"] == "1:1"
    audio = [s for s in data["streams"] if s["codec_type"] == "audio"]
    if mode == "silent_portrait":
        assert not audio
    else:
        assert audio[0]["codec_name"] == "aac"
    # Check both streams can be decoded all the way through.
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
                   check=True, capture_output=True, text=True)


def _probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _assert_timing(path, duration, pulses):
    """Check decoded flash/beep timing, not just container duration metadata."""
    data = _probe(path)
    video, audio = [next(s for s in data["streams"] if s["codec_type"] == kind)
                    for kind in ["video", "audio"]]
    for stream in [video, audio]:
        assert float(stream["start_time"]) == pytest.approx(0, abs=1 / 30)
        assert float(stream["duration"]) == pytest.approx(duration, abs=1 / 30)
    assert float(data["format"]["duration"]) == pytest.approx(duration, abs=1 / 30)
    frame_rate = float(Fraction(video["avg_frame_rate"]))
    frames = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=1:1,format=gray",
        "-an", "-f", "rawvideo", "-",
    ], check=True, capture_output=True).stdout
    flash_times = np.flatnonzero(np.frombuffer(frames, dtype=np.uint8) > 220) / frame_rate
    samples = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ar", "48000", "-ac", "1",
        "-f", "f32le", "-",
    ], check=True, capture_output=True).stdout
    signal = np.frombuffer(samples, dtype="<f4")
    # AAC may add a partial final packet; evaluate the expected interval only.
    bins = signal[:int(duration * 48000) // 480 * 480].reshape(-1, 480)
    audio_times = np.flatnonzero(np.sqrt(np.mean(bins ** 2, axis=1)) > 0.1) / 100
    for timestamp in pulses:
        flash = flash_times[np.abs(flash_times - timestamp) < 0.2]
        beep = audio_times[np.abs(audio_times - timestamp) < 0.2]
        assert len(flash) and len(beep)
        assert flash[0] == pytest.approx(timestamp, abs=1 / 30 + 0.01)
        # Allow one 30 fps frame for AAC packetization and 10 ms RMS bins.
        assert beep[0] == pytest.approx(timestamp, abs=1 / 30)
        assert flash[0] == pytest.approx(beep[0], abs=1 / 30 + 0.01)
    assert len(frames) == pytest.approx(duration * frame_rate, abs=1)


@pytest.mark.ffmpeg
@pytest.mark.parametrize("stabilize", [False, True])
@pytest.mark.parametrize("selection", ["single", "cross_file", "later_file"])
def test_seek_preserves_content_and_av_sync(ffmpeg_sources, tmp_path, selection, stabilize):
    output = tmp_path / "timed.mp4"
    options = dict(compress=True, stabilize=stabilize, stabilize_smoothing=5)
    if selection == "single":
        # Fractional, non-keyframe seek checks rounding and stream alignment.
        cut_video(ffmpeg_sources[0], output, 2.217, 3.717, **options)
        duration, pulses = 1.5, [2.5 - 2.217, 3.5 - 2.217]
    elif selection == "cross_file":
        concat_and_cut(ffmpeg_sources, output, 3, 6, **options)
        duration, pulses = 3, [0.5, 2.5]
        # Verify the join contains the first file followed by the second.
        rgb = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(output),
            "-vf", "scale=1:1,format=rgb24", "-an", "-f", "rawvideo", "-",
        ], check=True, capture_output=True).stdout
        pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(-1, 3)
        assert pixels[0, 0] > 200 and pixels[0, 2] < 50  # red
        assert pixels[45, 2] > 200 and pixels[45, 0] < 50  # blue after join
    else:
        concat_and_cut(ffmpeg_sources, output, 5.217, 6.717, **options)
        duration, pulses = 1.5, [5.5 - 5.217, 6.5 - 5.217]
    _assert_timing(output, duration, pulses)


@pytest.mark.ffmpeg
@pytest.mark.parametrize("stabilize", [False, True])
@pytest.mark.parametrize("size,sar,expected", [
    ("320x180", "1", (320, 180)),
    ("2560x1440", "1", (1920, 1080)),
    ("1280x1440", "1", (960, 1080)),
    ("319x181", "1", (318, 180)),
    ("720x576", "16/15", (720, 576)),
])
def test_scale_without_upscaling(ffmpeg_sources, tmp_path, size, sar, expected, stabilize):
    source, output = tmp_path / "source.mp4", tmp_path / "scaled.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"testsrc=size={size}:rate=30,format=yuv444p,setsar={sar}",
        "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", str(source),
    ], check=True, capture_output=True, text=True)
    cut_video(source, output, 0.2, 0.7, compress=True, stabilize=stabilize,
              stabilize_smoothing=5)
    video = next(s for s in _probe(output)["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == expected
    assert video["width"] % 2 == video["height"] % 2 == 0
    width, height = map(int, size.split("x"))
    expected_dar = width / height * float(Fraction(sar))
    actual_dar = video["width"] / video["height"] * float(Fraction(video["sample_aspect_ratio"].replace(":", "/")))
    assert actual_dar == pytest.approx(expected_dar, rel=0.001)
    assert float(video["duration"]) == pytest.approx(0.5, abs=1 / 30)


@pytest.mark.ffmpeg
@pytest.mark.parametrize("stabilize", [False, True])
def test_silent_concat(ffmpeg_sources, tmp_path, stabilize):
    source, output = tmp_path / "silent.mp4", tmp_path / "silent-cut.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(ffmpeg_sources[0]),
        "-an", "-c:v", "copy", str(source),
    ], check=True, capture_output=True, text=True)
    concat_and_cut([source, source], output, 3.5, 4.5, compress=True,
                   stabilize=stabilize, stabilize_smoothing=5)
    data = _probe(output)
    assert all(s["codec_type"] != "audio" for s in data["streams"])
    assert float(data["format"]["duration"]) == pytest.approx(1, abs=1 / 30)


@pytest.mark.ffmpeg
def test_long_offset_120_to_180(ffmpeg_sources, tmp_path):
    # Tiny frames keep this literal 120–180 s regression inexpensive.
    paths = [tmp_path / "long-first.mp4", tmp_path / "long-second.mp4"]
    for path, length, color, pulse in zip(paths, [150, 60], ["red", "blue"], [120.5, 10.5]):
        interval = f"between(t,{pulse},{pulse + 0.1})"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            f"color={color}:size=64x48:rate=30,drawbox=color=white:t=fill:enable='{interval}'",
            "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*880*t)*{interval}':s=48000",
            "-t", str(length), "-c:v", "libx264", "-preset", "ultrafast",
            "-bf", "2", "-g", "60", "-c:a", "aac", str(path),
        ], check=True, capture_output=True, text=True)
    output = tmp_path / "minute.mp4"
    concat_and_cut(paths, output, 120, 180, compress=True)
    _assert_timing(output, 60, [0.5, 40.5])
