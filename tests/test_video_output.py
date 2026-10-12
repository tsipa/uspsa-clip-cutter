"""Encoding contracts and real ffmpeg checks for compressed/stabilized clips."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from threading import Lock
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
    probe = Mock()
    monkeypatch.setattr(ffmpeg_utils, "get_frame_rate", probe)
    cut_video(tmp_path / "input.mp4", tmp_path / "out.mp4", 5, 5 + duration,
              accurate=False, compress=True)
    # One CRF encode for any length; no size target or second encode pass.
    render.assert_called_once()
    probe.assert_not_called()
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
    monkeypatch.setattr(ffmpeg_utils, "get_frame_rate", Mock(return_value=60.0))
    concat_and_cut([tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path / "out.mp4",
                   55, 70, accurate=False, compress=True, stabilize=True,
                   stabilize_smoothing=15)
    assert render.call_count == 2
    detection, encode = [call.args[0] for call in render.call_args_list]
    detection_filters = detection[detection.index("-vf") + 1]
    encode_filters = encode[encode.index("-vf") + 1]
    assert detection_filters.split(",vidstabdetect")[0] == encode_filters.split(",vidstabtransform")[0]
    assert "smoothing=15:" in encode_filters
    assert "fps=" not in detection_filters
    assert encode_filters.endswith(",fps=30")
    assert "optzoom=1:zoom=0:crop=black" in encode_filters
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


@pytest.mark.parametrize("fps,radius", [
    (24, 10), (25, 10), (30000 / 1001, 10), (30, 10),
    (50, 10), (60000 / 1001, 10), (60, 10), (120, 10),
    (60, 15), (60, 0), (60, 1000),
])
def test_smoothing_uses_literal_source_frames(tmp_path, monkeypatch, fps, radius):
    render = Mock()
    monkeypatch.setattr(ffmpeg_utils, "_run_render", render)
    monkeypatch.setattr(ffmpeg_utils, "get_frame_rate", Mock(return_value=fps))
    cut_video(tmp_path / "input.mp4", tmp_path / "out.mp4", 5, 7,
              compress=True, stabilize=True, stabilize_smoothing=radius)
    encode = render.call_args.args[0]
    assert f"smoothing={radius}:" in encode[encode.index("-vf") + 1]


@pytest.mark.parametrize("average,nominal,expected", [
    ("60/1", "120/1", 60), ("60000/1001", "60/1", 60000 / 1001),
    ("0/0", "30000/1001", 30000 / 1001), ("0/1", "25/1", 25),
])
def test_frame_rate_probe(tmp_path, monkeypatch, average, nominal, expected):
    result = subprocess.CompletedProcess([], 0, stdout=json.dumps({"streams": [
        {"avg_frame_rate": average, "r_frame_rate": nominal},
    ]}))
    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", Mock(return_value=result))
    assert ffmpeg_utils.get_frame_rate(tmp_path / "input.mp4") == pytest.approx(expected)


def test_frame_rate_probe_rejects_missing_rate(tmp_path, monkeypatch):
    result = subprocess.CompletedProcess([], 0, stdout='{"streams": [{"avg_frame_rate": "0/0"}]}')
    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", Mock(return_value=result))
    with pytest.raises(RuntimeError, match="Cannot determine video frame rate"):
        ffmpeg_utils.get_frame_rate(tmp_path / "input.mp4")


def test_concat_probes_rate_of_file_at_clip_start(tmp_path, monkeypatch):
    paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    monkeypatch.setattr(ffmpeg_utils, "get_duration", Mock(return_value=60.0))
    probe = Mock(return_value=60)
    monkeypatch.setattr(ffmpeg_utils, "get_frame_rate", probe)
    monkeypatch.setattr(ffmpeg_utils, "_run_render", Mock())
    concat_and_cut(paths, tmp_path / "out.mp4", 60, 70, compress=True, stabilize=True)
    probe.assert_called_once_with(paths[1])


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
    check.assert_called_once_with(compress=True, stabilize=True, gpu_encode=False)


@pytest.mark.parametrize("option,value", [("--compress-crf", "52"), ("--stabilize-smoothing", "-1"), ("--gpu-cq", "0"), ("--gpu-cq", "52")])
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
                                         "--compress", "--stabilize", "--gpu-encode", "--dry-run"])
    assert result.exit_code == 0, result.output
    check.assert_not_called()


def test_missing_features_fail_with_actionable_error(monkeypatch):
    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", Mock(return_value=
                        subprocess.CompletedProcess([], 0, stdout=" V..... libx264 H.264\n")))
    with pytest.raises(ffmpeg_utils.FFmpegFeatureError, match="libx265"):
        ffmpeg_utils.check_ffmpeg_features(compress=True)


@pytest.mark.parametrize("cross_file", [False, True])
@pytest.mark.parametrize("gpu_encode", [False, True])
@pytest.mark.parametrize("video_workers", [1, 2])
def test_pipeline_passes_output_options(tmp_path, monkeypatch, cross_file, gpu_encode, video_workers):
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
                                       compress_crf=28, stabilize=True, stabilize_smoothing=15,
                                       gpu_encode=gpu_encode, gpu_cq=29, video_workers=video_workers)
    rows = pipeline._cut_stages([stage], files, tmp_path / "out", config, tmp_path)
    assert rows[0].status == "ok"
    assert cut.call_args.kwargs["compress"] and cut.call_args.kwargs["stabilize"]
    assert cut.call_args.kwargs["compress_crf"] == 28
    assert cut.call_args.kwargs["stabilize_smoothing"] == 15
    assert cut.call_args.kwargs["gpu_encode"] == gpu_encode
    assert cut.call_args.kwargs["gpu_cq"] == 29


def test_gpu_cli_enables_compression_independently_of_whisper(tmp_path, monkeypatch):
    check, batch = Mock(), Mock(return_value=[])
    monkeypatch.setattr(cli, "check_ffmpeg_available", Mock())
    monkeypatch.setattr(cli, "check_ffmpeg_features", check)
    monkeypatch.setattr(cli, "run_batch", batch)
    monkeypatch.setattr(cli, "write_manifest", Mock())
    result = CliRunner().invoke(cli.app, [str(tmp_path), str(tmp_path / "out"),
                                         "--gpu-encode", "--gpu-cq", "29", "--fast-cut"])
    assert result.exit_code == 0, result.output
    config = batch.call_args.args[2]
    assert config.gpu_encode and config.compress and config.accurate_cut
    assert config.gpu_cq == 29 and config.device == "cpu"
    assert config.stabilize_smoothing == 10
    check.assert_called_once_with(compress=True, stabilize=False, gpu_encode=True)


@pytest.mark.parametrize("cross_file", [False, True])
def test_gpu_encoding_options_and_profile(tmp_path, monkeypatch, cross_file):
    render = Mock()
    monkeypatch.setattr(ffmpeg_utils, "_run_render", render)
    monkeypatch.setattr(ffmpeg_utils, "get_duration", Mock(return_value=60.0))
    if cross_file:
        concat_and_cut([tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path / "out.mp4",
                       55, 65, accurate=False, gpu_encode=True, gpu_cq=29)
    else:
        cut_video(tmp_path / "a.mp4", tmp_path / "out.mp4", 5, 15,
                  accurate=False, gpu_encode=True, gpu_cq=29)
    render.assert_called_once()
    cmd = render.call_args.args[0]
    for option, expected in [("-c:v", "hevc_nvenc"), ("-preset", "p7"), ("-tune", "hq"),
                             ("-rc", "vbr"), ("-cq", "29"), ("-b:v", "0"),
                             ("-multipass", "fullres"), ("-rc-lookahead", "32"),
                             ("-spatial-aq", "1"), ("-bf", "0"),
                             ("-c:a", "aac"), ("-b:a", "96k")]:
        assert cmd[cmd.index(option) + 1] == expected
    assert "fps=30" in cmd[cmd.index("-vf") + 1]
    assert not {"-crf", "-temporal-aq", "-fs"}.intersection(cmd)


@pytest.mark.parametrize("runtime_ok", [False, True])
def test_gpu_preflight_checks_actual_encoder(tmp_path, monkeypatch, runtime_ok):
    # A GPU-only build needs NVENC/AAC, not libx265; the listed encoder must
    # still be tested against a real device before transcription.
    listing = subprocess.CompletedProcess([], 0, stdout=" V..... hevc_nvenc HEVC\n A..... aac AAC\n")
    encoding = subprocess.CompletedProcess([], 0 if runtime_ok else 1, stderr="driver unavailable")
    run = Mock(side_effect=[listing, encoding])
    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", run)
    if runtime_ok:
        ffmpeg_utils.check_ffmpeg_features(gpu_encode=True)
    else:
        with pytest.raises(ffmpeg_utils.FFmpegFeatureError, match="driver unavailable"):
            ffmpeg_utils.check_ffmpeg_features(gpu_encode=True)
    assert "hevc_nvenc" in run.call_args.args[0]
    assert run.call_args.kwargs["timeout"] == 15


def test_gpu_preflight_timeout(monkeypatch):
    listing = subprocess.CompletedProcess([], 0, stdout=" V..... hevc_nvenc HEVC\n A..... aac AAC\n")
    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", Mock(side_effect=[
        listing, subprocess.TimeoutExpired("ffmpeg", 15),
    ]))
    with pytest.raises(ffmpeg_utils.FFmpegFeatureError, match="timed out"):
        ffmpeg_utils.check_ffmpeg_features(gpu_encode=True)


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


@pytest.fixture(scope="module")
def ffmpeg_sources_by_rate(ffmpeg_sources, tmp_path_factory):
    folder = tmp_path_factory.mktemp("30fps sources")
    sources_30 = [folder / path.name for path in ffmpeg_sources]
    for source, output in zip(ffmpeg_sources, sources_30):
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(source), "-vf", "fps=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-bf", "2", "-g", "60",
            "-c:a", "copy", str(output),
        ], check=True, capture_output=True, text=True)
    return {30: sources_30, 60: ffmpeg_sources}


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


@pytest.mark.ffmpeg
@pytest.mark.parametrize("source_rate", [30, 60])
@pytest.mark.parametrize("stabilize", [False, True])
@pytest.mark.parametrize("gpu_encode", [False, True])
def test_parallel_pipeline_preserves_timing_and_private_passes(
    ffmpeg_sources_by_rate, tmp_path, monkeypatch, request, source_rate, stabilize, gpu_encode,
):
    if gpu_encode:
        request.getfixturevalue("nvenc_available")
    files = [pipeline.FileInfo(
        path=path, wav_path=tmp_path / f"source{i}.wav", duration=4,
        creation_epoch=4 * i, creation_str=f"2026-10-11_12-00-0{i}",
        creation_iso="2026-10-11T12:00:00",
    ) for i, path in enumerate(ffmpeg_sources_by_rate[source_rate])]
    # Render a cross-file clip and a fractional single-file cut concurrently.
    stages = [pipeline.Stage(clip_start=3, clip_end=6, complete=True),
              pipeline.Stage(clip_start=5.217, clip_end=6.717, complete=True)]
    config = pipeline.ProcessingConfig(
        video_workers=2, compress=True, stabilize=stabilize, gpu_encode=gpu_encode,
        start_padding=0, end_padding=0, min_clip_length=0.1, stabilize_smoothing=5,
    )
    output, debug = tmp_path / "out", tmp_path / "debug"
    debug.mkdir()
    passes, lock = {}, Lock()
    render = ffmpeg_utils._run_render

    def record(cmd, cwd):
        with lock:
            passes.setdefault(cwd, []).append("vidstabdetect=" in " ".join(cmd))
        render(cmd, cwd)

    monkeypatch.setattr(ffmpeg_utils, "_run_render", record)
    rows = pipeline._cut_stages(stages, files, output, config, debug)
    assert [row.status for row in rows] == ["ok", "ok"], [row.error_message for row in rows]
    assert rows[0].source_file == " + ".join(file.path.name for file in files)
    assert rows[1].source_file == files[1].path.name
    _assert_timing(Path(rows[0].output_file), 3, [0.5, 2.5])
    _assert_timing(Path(rows[1].output_file), 1.5, [5.5 - 5.217, 6.5 - 5.217])
    assert len(passes) == 2
    assert all(phases == ([True, False] if stabilize else [False]) for phases in passes.values())
    assert all(not cwd.exists() for cwd in passes)


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
    assert video["avg_frame_rate"] == "30/1"
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
@pytest.mark.parametrize("source_rate", [30, 60])
@pytest.mark.parametrize("stabilize", [False, True])
@pytest.mark.parametrize("selection", ["single", "cross_file", "later_file"])
def test_seek_preserves_content_and_av_sync(ffmpeg_sources_by_rate, tmp_path, selection, stabilize, source_rate):
    ffmpeg_sources = ffmpeg_sources_by_rate[source_rate]
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
@pytest.mark.parametrize("source_rate", [30, 60])
@pytest.mark.parametrize("large_source", [False, True])
def test_stabilization_passes_receive_identical_original_frames(
    ffmpeg_sources_by_rate, tmp_path, monkeypatch, source_rate, large_source,
):
    records = {}
    commands = {}

    def inspect_pass(cmd, cwd):
        cmd = cmd.copy()
        index = cmd.index("-vf") + 1
        stage = "vidstabdetect" if "vidstabdetect" in cmd[index] else "vidstabtransform"
        commands[stage] = cmd[index]
        # Inspect frames immediately before either stabilization filter. showinfo
        # does not change frames; PTS and checksums prove order and pixel identity.
        cmd[index] = cmd[index].replace(stage, "showinfo," + stage)
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        records[stage] = re.findall(
            r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:.*?s:(\d+x\d+).*?checksum:([0-9A-F]+)",
            result.stderr,
        )

    monkeypatch.setattr(ffmpeg_utils, "_run_render", inspect_pass)
    output = tmp_path / "original-rate.mp4"
    if large_source:
        source = tmp_path / "large-444.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            f"testsrc=size=2560x1440:rate={source_rate},format=yuv444p",
            "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", str(source),
        ], check=True, capture_output=True, text=True)
        cut_video(source, output, 0.2, 0.7, compress=True, stabilize=True)
        duration, dimensions = 0.5, "1920x1080"
    else:
        concat_and_cut(ffmpeg_sources_by_rate[source_rate], output, 3, 6,
                       compress=True, stabilize=True)
        duration, dimensions = 3, "320x180"
    assert records["vidstabdetect"] == records["vidstabtransform"]
    assert len(records["vidstabdetect"]) == round(duration * source_rate)
    assert all(frame[2] == dimensions for frame in records["vidstabdetect"])
    assert "fps=" not in commands["vidstabdetect"]
    assert commands["vidstabtransform"].endswith(",fps=30")
    assert "smoothing=10:optzoom=1:zoom=0:crop=black:interpol=bicubic" in commands["vidstabtransform"]
    video = next(s for s in _probe(output)["streams"] if s["codec_type"] == "video")
    assert video["avg_frame_rate"] == "30/1"
    assert video["codec_name"] == "hevc"


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


@pytest.fixture(scope="module")
def nvenc_available(ffmpeg_sources):
    try:
        ffmpeg_utils.check_ffmpeg_features(gpu_encode=True)
    except ffmpeg_utils.FFmpegFeatureError as exc:
        pytest.skip(str(exc))


@pytest.mark.ffmpeg
@pytest.mark.parametrize("stabilize", [False, True])
@pytest.mark.parametrize("mode", ["single", "concat", "silent", "large"])
def test_real_nvenc_output(nvenc_available, ffmpeg_sources, tmp_path, mode, stabilize):
    output = tmp_path / "gpu.mp4"
    options = dict(gpu_encode=True, stabilize=stabilize, stabilize_smoothing=5)
    if mode == "single":
        cut_video(ffmpeg_sources[0], output, 2.217, 3.717, **options)
        duration, dimensions = 1.5, (320, 180)
        _assert_timing(output, duration, [2.5 - 2.217, 3.5 - 2.217])
    elif mode == "concat":
        concat_and_cut(ffmpeg_sources, output, 3, 6, **options)
        duration, dimensions = 3, (320, 180)
        _assert_timing(output, duration, [0.5, 2.5])
    elif mode == "silent":
        source = tmp_path / "silent.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(ffmpeg_sources[0]),
            "-an", "-c:v", "copy", str(source),
        ], check=True, capture_output=True, text=True)
        concat_and_cut([source, source], output, 3.5, 4.5, **options)
        duration, dimensions = 1, (320, 180)
    else:
        source = tmp_path / "large.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc=size=2560x1440:rate=60,format=yuv444p", "-t", "1",
            "-c:v", "libx264", "-preset", "ultrafast", str(source),
        ], check=True, capture_output=True, text=True)
        cut_video(source, output, 0.2, 0.7, **options)
        duration, dimensions = 0.5, (1920, 1080)
    data = _probe(output)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert video["codec_name"] == "hevc"
    assert video["codec_tag_string"] == "hvc1"
    assert video["avg_frame_rate"] == "30/1"
    assert video["pix_fmt"] == "yuv420p"
    assert (video["width"], video["height"]) == dimensions
    assert float(video["duration"]) == pytest.approx(duration, abs=1 / 30)
    audio = [s for s in data["streams"] if s["codec_type"] == "audio"]
    if mode in {"silent", "large"}:
        assert not audio
    else:
        assert audio[0]["codec_name"] == "aac"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
                   check=True, capture_output=True, text=True)
