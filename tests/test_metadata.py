"""Tests for metadata module."""

from datetime import datetime, timezone
from unittest.mock import patch

from video_stage_cutter.metadata import _extract_creation_time


class TestExtractCreationTime:
    def test_format_tags(self) -> None:
        probe = {
            "format": {"tags": {"creation_time": "2024-06-15T14:30:00.000000Z"}},
            "streams": [],
        }
        dt = _extract_creation_time(probe)
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15
        assert dt.tzinfo is not None

    def test_stream_tags_fallback(self) -> None:
        probe = {
            "format": {"tags": {}},
            "streams": [
                {"tags": {"creation_time": "2023-01-01T00:00:00Z"}},
            ],
        }
        dt = _extract_creation_time(probe)
        assert dt is not None
        assert dt.year == 2023

    def test_no_tags_returns_none(self) -> None:
        probe = {"format": {"tags": {}}, "streams": []}
        dt = _extract_creation_time(probe)
        assert dt is None

    def test_apple_quicktime_tag(self) -> None:
        probe = {
            "format": {"tags": {"com.apple.quicktime.creationdate": "2024-03-10T09:15:00.000000Z"}},
            "streams": [],
        }
        dt = _extract_creation_time(probe)
        assert dt is not None
        assert dt.month == 3

    def test_timezone_offset_format(self) -> None:
        probe = {
            "format": {"tags": {"creation_time": "2024-08-20T10:00:00+0500"}},
            "streams": [],
        }
        dt = _extract_creation_time(probe)
        assert dt is not None
        assert dt.tzinfo is not None
