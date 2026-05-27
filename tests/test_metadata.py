"""Tests for metadata module."""

from pathlib import Path

from video_stage_cutter.metadata import _extract_creation_time, filename_sort_key


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


class TestFilenameSortKey:
    def test_gopro_chapters_sorted_correctly(self) -> None:
        files = [
            Path("GX010124.mp4"),
            Path("GX020123.mp4"),
            Path("GX010123.mp4"),
        ]
        result = sorted(files, key=filename_sort_key)
        assert [f.name for f in result] == [
            "GX010123.mp4",  # clip 123, chapter 1
            "GX020123.mp4",  # clip 123, chapter 2
            "GX010124.mp4",  # clip 124, chapter 1
        ]

    def test_gopro_old_naming(self) -> None:
        files = [
            Path("GP010001.mp4"),
            Path("GOPR0001.mp4"),
            Path("GOPR0002.mp4"),
        ]
        result = sorted(files, key=filename_sort_key)
        assert [f.name for f in result] == [
            "GOPR0001.mp4",  # clip 1, chapter 0
            "GP010001.mp4",  # clip 1, chapter 1
            "GOPR0002.mp4",  # clip 2, chapter 0
        ]

    def test_non_gopro_lexicographic(self) -> None:
        files = [
            Path("DJI_0003.mp4"),
            Path("DJI_0001.mp4"),
            Path("DJI_0002.mp4"),
        ]
        result = sorted(files, key=filename_sort_key)
        assert [f.name for f in result] == [
            "DJI_0001.mp4",
            "DJI_0002.mp4",
            "DJI_0003.mp4",
        ]

    def test_mixed_gopro_and_other(self) -> None:
        files = [
            Path("video_003.mp4"),
            Path("GX010001.mp4"),
            Path("video_001.mp4"),
        ]
        result = sorted(files, key=filename_sort_key)
        # GoPro files (prefix 0) come before non-GoPro (prefix 1)
        assert result[0].name == "GX010001.mp4"
