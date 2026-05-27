"""CSV manifest writer for batch results."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)

FIELDNAMES = [
    "source_file",
    "creation_time",
    "duration",
    "start_offset",
    "end_offset",
    "start_reason",
    "end_reason",
    "confidence",
    "output_file",
    "status",
    "error_message",
]


@dataclass
class ManifestRow:
    source_file: str = ""
    creation_time: str = ""
    duration: str = ""
    start_offset: str = ""
    end_offset: str = ""
    start_reason: str = ""
    end_reason: str = ""
    confidence: str = ""
    output_file: str = ""
    status: str = ""
    error_message: str = ""


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write (or overwrite) the manifest CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({f.name: getattr(row, f.name) for f in fields(row)})
    log.info("Wrote manifest with %d rows to %s", len(rows), path)
