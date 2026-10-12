"""Sleep prevention is scoped to processing and always unwound."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from typer.testing import CliRunner

from video_stage_cutter import cli, power


@pytest.fixture
def windows_state(monkeypatch):
    monkeypatch.setattr(power, "sys", SimpleNamespace(platform="win32"))
    set_state = Mock(side_effect=[power.ES_CONTINUOUS, power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED])
    loader = Mock(return_value=SimpleNamespace(SetThreadExecutionState=set_state))
    monkeypatch.setattr(power.ctypes, "WinDLL", loader, raising=False)
    return set_state


@pytest.mark.parametrize("exception", [None, RuntimeError, KeyboardInterrupt])
def test_power_request_is_released(windows_state, exception):
    def processing():
        with power.prevent_sleep():
            assert windows_state.call_count == 1
            if exception:
                raise exception()
    if exception:
        with pytest.raises(exception):
            processing()
    else:
        processing()
    assert windows_state.call_args_list == [
        call(power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED), call(power.ES_CONTINUOUS),
    ]


def test_power_restores_previous_requirements(windows_state):
    previous = power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED
    windows_state.side_effect = [previous, previous]
    with power.prevent_sleep():
        pass
    assert windows_state.call_args == call(previous)


def test_non_windows_does_not_load_kernel(monkeypatch):
    monkeypatch.setattr(power, "sys", SimpleNamespace(platform="linux"))
    loader = Mock(side_effect=AssertionError("Windows DLL on Linux"))
    monkeypatch.setattr(power.ctypes, "WinDLL", loader, raising=False)
    with power.prevent_sleep():
        pass
    loader.assert_not_called()


def test_rejected_power_request_logs_warning(windows_state, caplog):
    windows_state.side_effect = [0]
    with power.prevent_sleep():
        pass
    assert windows_state.call_count == 1
    assert "rejected idle-sleep prevention" in caplog.text


def test_cleanup_error_preserves_processing_error(windows_state, caplog):
    windows_state.side_effect = [power.ES_CONTINUOUS, OSError("cleanup failed")]
    with pytest.raises(RuntimeError, match="processing error"):
        with power.prevent_sleep():
            raise RuntimeError("processing error")
    assert "cleanup failed" in caplog.text


@pytest.mark.parametrize("exception", [None, KeyboardInterrupt])
def test_cli_holds_request_through_manifest(tmp_path, monkeypatch, exception):
    events = []

    @contextmanager
    def guard():
        events.append("acquire")
        try:
            yield
        finally:
            events.append("release")

    def processing(*args):
        events.append("processing")
        if exception:
            raise exception()
        return []

    monkeypatch.setattr(cli, "prevent_sleep", guard)
    monkeypatch.setattr(cli, "check_ffmpeg_available", Mock())
    monkeypatch.setattr(cli, "check_ffmpeg_features", Mock())
    monkeypatch.setattr(cli, "run_batch", processing)
    monkeypatch.setattr(cli, "write_manifest", lambda *args: events.append("manifest"))
    result = CliRunner().invoke(cli.app, [str(tmp_path), str(tmp_path / "out")])
    if exception:
        assert result.exit_code != 0
        assert events == ["acquire", "processing", "release"]
    else:
        assert result.exit_code == 0, result.output
        assert events == ["acquire", "processing", "manifest", "release"]
