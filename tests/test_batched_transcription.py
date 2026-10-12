"""Batched transcription selection, audio coverage, timestamps and model reuse."""

from __future__ import annotations

import sys
import wave
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from video_stage_cutter import cli, pipeline
from video_stage_cutter.phrase_detect import detect_phrases
from video_stage_cutter.transcribe import load_transcript, save_transcript


def _wav(path, duration, sample_rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * int(duration * sample_rate))
    return path


def _decoded_segments():
    words = [SimpleNamespace(start=31.1, end=31.3, word="stand", probability=0.9),
             SimpleNamespace(start=31.3, end=31.5, word="by", probability=0.8)]
    end_text = "if you are finished unload and show clear if clear hammer down and holster range is clear"
    end_words = [SimpleNamespace(start=60.5 + i * 0.1, end=60.6 + i * 0.1,
                                 word=word, probability=0.9)
                 for i, word in enumerate(end_text.split())]
    return [
        SimpleNamespace(start=31.1, end=31.5, text=" stand by ", words=words, seek=3000),
        SimpleNamespace(start=60.5, end=62.2, text=end_text, words=end_words, seek=6000),
        SimpleNamespace(start=63, end=64, text="range noise", words=None, seek=6000),
    ]


@pytest.fixture
def whisper_backend(monkeypatch):
    model, batch = Mock(), Mock()
    model.frames_per_second = 100
    batch.model = model
    for runner in [model, batch]:
        runner.transcribe.side_effect = lambda *a, **kw: (iter(_decoded_segments()), None)
    module = SimpleNamespace(WhisperModel=Mock(return_value=model),
                             BatchedInferencePipeline=Mock(return_value=batch))
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


@pytest.mark.parametrize("duration", [0, 0.25, 30, 30.25, 60, 60.0625, 95])
@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_batched_chunks_cover_every_sample(tmp_path, duration, sample_rate):
    path = _wav(tmp_path / "audio.wav", duration, sample_rate)
    chunks = pipeline._batched_clip_timestamps(path)
    if not duration:
        assert chunks == []
        return
    assert chunks[0]["start"] == 0
    assert chunks[-1]["end"] == duration
    assert all(0 < c["end"] - c["start"] <= 30 for c in chunks)
    assert all(a["end"] == b["start"] for a, b in zip(chunks, chunks[1:]))
    assert sum(c["end"] - c["start"] for c in chunks) == duration


def test_batched_reuses_one_pipeline_and_preserves_phrase_times(tmp_path, whisper_backend):
    first = _wav(tmp_path / "first.wav", 65)
    second = _wav(tmp_path / "second.wav", 95)
    config = pipeline.ProcessingConfig(device="cuda", batched=True, batch_size=2)
    segments, cached = pipeline._transcribe(first, config, None)
    again, same_cached = pipeline._transcribe(second, config, cached)
    whisper_backend.WhisperModel.assert_called_once_with("large-v3", device="cuda", compute_type="int8")
    model = whisper_backend.WhisperModel.return_value
    whisper_backend.BatchedInferencePipeline.assert_called_once_with(model=model)
    model.transcribe.assert_not_called()
    assert cached is same_cached is whisper_backend.BatchedInferencePipeline.return_value
    calls = cached.transcribe.call_args_list
    assert len(calls) == 2
    for call, last_end in zip(calls, [65, 95]):
        options = call.kwargs
        assert options["batch_size"] == 2
        assert options["vad_filter"] is False
        assert options["chunk_length"] == 30
        assert options["condition_on_previous_text"] is False
        assert options["beam_size"] == 5 and options["word_timestamps"] is True
        assert options["language"] == "en"
        assert options["initial_prompt"] == pipeline._WHISPER_PROMPT
        assert options["clip_timestamps"][-1]["end"] == last_end
    assert segments == again
    assert segments[0].text == "stand by"
    assert segments[0].words[0].start == 31.1
    assert segments[1].start == 60.5 and segments[2].words == []
    starts, ends = detect_phrases(segments)
    assert starts[0].start == 31.1
    assert ends[0].start >= 60.5
    path = tmp_path / "transcript.json"
    save_transcript(segments, path)
    assert load_transcript(path) == segments


def test_normal_transcription_keeps_existing_options(tmp_path, whisper_backend):
    config = pipeline.ProcessingConfig(batch_size=7)
    _, cached = pipeline._transcribe(tmp_path / "unused.wav", config, None)
    pipeline._transcribe(tmp_path / "second.wav", config, cached)
    whisper_backend.WhisperModel.assert_called_once_with("large-v3", device="cpu", compute_type="int8")
    whisper_backend.BatchedInferencePipeline.assert_not_called()
    options = cached.transcribe.call_args.kwargs
    assert options == dict(beam_size=5, word_timestamps=True, language="en",
                           initial_prompt=pipeline._WHISPER_PROMPT, vad_filter=False)


@pytest.mark.parametrize("batched", [False, True])
def test_worker_uses_selected_transcription_mode(tmp_path, monkeypatch, whisper_backend, batched):
    video, wav = tmp_path / "source.mp4", _wav(tmp_path / "audio.wav", 65)
    video.write_bytes(b"video")
    extract = Mock()
    monkeypatch.setattr(pipeline, "extract_audio", extract)
    index, segments = pipeline._transcribe_worker(
        (3, str(video), str(wav), str(tmp_path), "large-v3", "cuda", "int8", False, False, batched, 3),
    )
    assert index == 3
    extract.assert_called_once_with(video, wav)
    assert load_transcript(tmp_path / "source_transcript.json") == segments
    if batched:
        assert whisper_backend.BatchedInferencePipeline.return_value.transcribe.call_args.kwargs["batch_size"] == 3
    else:
        whisper_backend.BatchedInferencePipeline.assert_not_called()


def test_reused_transcript_bypasses_batched_model(tmp_path, monkeypatch, whisper_backend):
    config = pipeline.ProcessingConfig(batched=True, reuse_transcripts=True)
    expected, _ = pipeline._transcribe(_wav(tmp_path / "initial.wav", 65), config, None)
    save_transcript(expected, tmp_path / "source_transcript.json")
    whisper_backend.WhisperModel.reset_mock()
    whisper_backend.BatchedInferencePipeline.reset_mock()
    fi = pipeline.FileInfo(path=tmp_path / "source.mp4", wav_path=tmp_path / "audio.wav",
                           duration=65, creation_epoch=0, creation_str="now", creation_iso="now")
    monkeypatch.setattr(pipeline, "extract_audio", Mock())
    pipeline._transcribe_file(fi, 0, config, tmp_path, None)
    assert fi.segments == expected
    pipeline._transcribe_worker((0, str(fi.path), str(fi.wav_path), str(tmp_path),
                                 "large-v3", "cuda", "int8", True, False, True, 2))
    whisper_backend.WhisperModel.assert_not_called()
    whisper_backend.BatchedInferencePipeline.assert_not_called()


def test_parallel_dispatch_passes_batch_options(tmp_path, monkeypatch):
    from concurrent import futures

    paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    monkeypatch.setattr(pipeline, "discover_videos", Mock(return_value=(paths, dict.fromkeys(paths))))
    monkeypatch.setattr(pipeline, "get_duration", Mock(return_value=65))
    monkeypatch.setattr(pipeline, "get_creation_time_or_mtime", Mock(return_value=datetime(2026, 10, 11)))
    for name in ["_detect_phrases_global", "_collect_audio_anchors_for_file", "_assemble_stages", "_cut_stages"]:
        monkeypatch.setattr(pipeline, name, Mock(return_value=[]))
    pool = Mock()
    pool.map.side_effect = lambda fn, args: [(a[0], []) for a in args]
    executor = Mock()
    executor.return_value.__enter__ = Mock(return_value=pool)
    executor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(futures, "ProcessPoolExecutor", executor)
    config = pipeline.ProcessingConfig(workers=2, batched=True, batch_size=3)
    pipeline.run_batch(tmp_path, tmp_path / "out", config)
    assert all(args[-2:] == (True, 3) for args in pool.map.call_args.args[1])


@pytest.mark.parametrize("flag", ["--batched", "--batched-transcription", None])
def test_cli_batch_options(tmp_path, monkeypatch, flag):
    batch = Mock(return_value=[])
    monkeypatch.setattr(cli, "check_ffmpeg_available", Mock())
    monkeypatch.setattr(cli, "check_ffmpeg_features", Mock())
    monkeypatch.setattr(cli, "run_batch", batch)
    monkeypatch.setattr(cli, "write_manifest", Mock())
    args = [str(tmp_path), str(tmp_path / "out"), "--device", "cuda", "--batch-size", "4"]
    if flag:
        args.append(flag)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    config = batch.call_args.args[2]
    assert config.batched == bool(flag)
    assert config.batch_size == 4 and config.workers == 1
    assert config.device == "cuda" and config.compute_type == "int8"


def test_cli_rejects_zero_batch_size(tmp_path):
    result = CliRunner().invoke(cli.app, [str(tmp_path), str(tmp_path / "out"), "--batched", "--batch-size", "0"])
    assert result.exit_code == 2
