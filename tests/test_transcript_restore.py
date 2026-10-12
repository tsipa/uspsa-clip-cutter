"""Durable transcription progress survives interruption and resumes safely."""

import json
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from video_stage_cutter import cli, pipeline, transcribe
from video_stage_cutter.transcribe import (
    TranscriptSegment, WordInfo, load_transcript, load_transcript_checkpoint,
    save_transcript, save_transcript_checkpoint, transcript_source,
)


def _segment(start, end, text, seek):
    return SimpleNamespace(start=start, end=end, text=text, seek=seek,
                           words=[SimpleNamespace(start=start, end=end, word=text, probability=0.9)])


@pytest.fixture
def recording(tmp_path):
    video, wav = tmp_path / "source.mp4", tmp_path / "source.wav"
    video.write_bytes(b"source video")
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 65 * 16000)
    return video, wav


@pytest.fixture
def backend(monkeypatch):
    import sys
    model, batched = Mock(), Mock()
    model.frames_per_second = 100
    batched.model = model
    module = SimpleNamespace(WhisperModel=Mock(return_value=model),
                             BatchedInferencePipeline=Mock(return_value=batched))
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return model, batched


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_interrupt_then_restore_replays_only_last_window(recording, tmp_path, monkeypatch, backend, worker, batched):
    video, wav = recording
    monkeypatch.setattr(pipeline, "extract_audio", Mock())
    runner = backend[1 if batched else 0]

    def interrupted():
        yield _segment(1, 2, "prefix", 0)
        yield _segment(31, 32, "partial_one", 3000)
        yield _segment(33, 34, "partial_two", 3000)
        raise KeyboardInterrupt()

    remaining = [_segment(31, 32, "replayed_one", 3000),
                 _segment(33, 34, "replayed_two", 3000),
                 _segment(61, 62, "tail", 6000)]
    runner.transcribe.side_effect = [(interrupted(), None), (iter(remaining), None)]
    config = pipeline.ProcessingConfig(batched=batched, batch_size=2)
    fi = pipeline.FileInfo(path=video, wav_path=wav, duration=65, creation_epoch=0,
                           creation_str="now", creation_iso="now")

    def run(restore):
        config.restore = restore
        if worker:
            return pipeline._transcribe_worker((0, str(video), str(wav), str(tmp_path),
                                                config.model, "cpu", "int8", False, restore, batched, 2))[1]
        pipeline._transcribe_file(fi, 0, config, tmp_path, None)
        return fi.segments

    checkpoint = tmp_path / "source_transcript.partial.json"
    completed = tmp_path / "source_transcript.json"
    with pytest.raises(KeyboardInterrupt):
        run(False)
    saved, cursor = load_transcript_checkpoint(checkpoint, transcript_source(video, config.model))
    assert [s.text for s in saved] == ["prefix", "partial_one", "partial_two"]
    assert cursor == 30 and not completed.exists()
    result = run(True)
    assert [s.text for s in result] == ["prefix", "replayed_one", "replayed_two", "tail"]
    assert [s.words[0].start for s in result] == [1, 31, 33, 61]
    assert load_transcript(completed) == result
    assert not checkpoint.exists()
    options = runner.transcribe.call_args.kwargs
    if batched:
        assert options["clip_timestamps"] == [{"start": 30, "end": 60}, {"start": 60, "end": 65}]
    else:
        assert options["clip_timestamps"] == [30]


@pytest.mark.parametrize("worker", [False, True])
def test_restore_reuses_completed_legacy_transcript(recording, tmp_path, monkeypatch, backend, worker):
    video, wav = recording
    expected = [TranscriptSegment(1, 2, "saved", [WordInfo(1, 2, "saved", 0.9)])]
    save_transcript(expected, tmp_path / "source_transcript.json")
    extract = Mock(side_effect=AssertionError("should not re-extract completed transcript"))
    monkeypatch.setattr(pipeline, "extract_audio", extract)
    config = pipeline.ProcessingConfig(restore=True)
    if worker:
        _, restored = pipeline._transcribe_worker((0, str(video), str(wav), str(tmp_path),
                                                   "large-v3", "cpu", "int8", False, True, False, 2))
    else:
        fi = pipeline.FileInfo(video, wav, 65, 0, "now", "now")
        pipeline._transcribe_file(fi, 0, config, tmp_path, None)
        restored = fi.segments
    assert restored == expected
    assert not backend[0].transcribe.called and not backend[1].transcribe.called


@pytest.mark.parametrize("damage", ["json", "source", "model", "cursor"])
def test_invalid_checkpoint_restarts_recognition(recording, tmp_path, backend, damage, caplog):
    video, wav = recording
    config = pipeline.ProcessingConfig(restore=True)
    checkpoint = tmp_path / "partial.json"
    old = [TranscriptSegment(1, 2, "old", [])]
    source = transcript_source(video, config.model)
    save_transcript_checkpoint(checkpoint, old, source, 30 if damage != "cursor" else 999)
    if damage == "json":
        checkpoint.write_text("{broken", encoding="utf-8")
    elif damage == "source":
        video.write_bytes(b"changed source video")
    elif damage == "model":
        config.model = "small"
    backend[0].transcribe.return_value = (iter([_segment(3, 4, "fresh", 0)]), None)
    segments, _ = pipeline._transcribe(wav, config, None, checkpoint_path=checkpoint, video_path=video)
    assert [s.text for s in segments] == ["fresh"]
    assert "clip_timestamps" not in backend[0].transcribe.call_args.kwargs
    assert "Ignoring invalid checkpoint" in caplog.text


@pytest.mark.parametrize("exception", [OSError, KeyboardInterrupt])
def test_failed_atomic_replace_preserves_previous_checkpoint(recording, tmp_path, monkeypatch, exception):
    video, _ = recording
    path = tmp_path / "state.json"
    source = transcript_source(video, "large-v3")
    original = [TranscriptSegment(1, 2, "saved", [])]
    save_transcript_checkpoint(path, original, source, 0)
    monkeypatch.setattr(transcribe.os, "replace", Mock(side_effect=exception("interrupted write")))
    with pytest.raises(exception):
        save_transcript_checkpoint(path, [TranscriptSegment(3, 4, "new", [])], source, 30)
    assert load_transcript_checkpoint(path, source) == (original, 0)
    assert list(tmp_path.glob(".*.tmp")) == []


def test_partial_is_preferred_over_older_completed_file(recording, tmp_path, monkeypatch):
    video, wav = recording
    save_transcript([TranscriptSegment(1, 2, "old complete", [])], tmp_path / "source_transcript.json")
    partial = tmp_path / "source_transcript.partial.json"
    partial.write_text("{}", encoding="utf-8")
    fresh = [TranscriptSegment(1, 2, "fresh", [])]
    monkeypatch.setattr(pipeline, "extract_audio", Mock())
    transcription = Mock(return_value=(fresh, None))
    monkeypatch.setattr(pipeline, "_transcribe", transcription)
    fi = pipeline.FileInfo(video, wav, 65, 0, "now", "now")
    pipeline._transcribe_file(fi, 0, pipeline.ProcessingConfig(reuse_transcripts=True), tmp_path, None)
    assert fi.segments == fresh and not partial.exists()
    transcription.assert_called_once()


def test_batched_resume_chunks_preserve_absolute_offsets(recording):
    _, wav = recording
    assert pipeline._batched_clip_timestamps(wav, 27) == [
        {"start": 27, "end": 57}, {"start": 57, "end": 65},
    ]


def test_cli_restore_option(tmp_path, monkeypatch):
    batch = Mock(return_value=[])
    monkeypatch.setattr(cli, "check_ffmpeg_available", Mock())
    monkeypatch.setattr(cli, "check_ffmpeg_features", Mock())
    monkeypatch.setattr(cli, "run_batch", batch)
    monkeypatch.setattr(cli, "write_manifest", Mock())
    result = CliRunner().invoke(cli.app, [str(tmp_path), str(tmp_path / "out"), "--restore", "--batched"])
    assert result.exit_code == 0, result.output
    assert batch.call_args.args[2].restore


def test_fresh_run_ignores_partial_without_restore(recording, tmp_path, backend):
    video, wav = recording
    path = tmp_path / "partial.json"
    save_transcript_checkpoint(path, [TranscriptSegment(1, 2, "old", [])],
                               transcript_source(video, "large-v3"), 30)
    backend[0].transcribe.return_value = (iter([_segment(3, 4, "fresh", 0)]), None)
    segments, _ = pipeline._transcribe(wav, pipeline.ProcessingConfig(), None,
                                       checkpoint_path=path, video_path=video)
    assert [s.text for s in segments] == ["fresh"]


def test_seek_rounding_keeps_previous_window_segment(recording, tmp_path, backend):
    video, wav = recording
    path = tmp_path / "partial.json"
    previous = TranscriptSegment(20, 30.005, "completed window", [])
    partial = TranscriptSegment(31, 32, "replay this", [])
    save_transcript_checkpoint(path, [previous, partial], transcript_source(video, "large-v3"), 30)
    backend[0].transcribe.return_value = (iter([_segment(31, 32, "replayed", 3000)]), None)
    segments, _ = pipeline._transcribe(wav, pipeline.ProcessingConfig(restore=True), None,
                                       checkpoint_path=path, video_path=video)
    assert segments[0] == previous
    assert [s.text for s in segments] == ["completed window", "replayed"]
