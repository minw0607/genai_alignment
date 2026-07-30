"""Artifact trail — the documentation-trail section every scenario notebook ends with.

Lists exactly what a run produced and where, with existence/size/modified-time
checked live at render time rather than just asserted in prose — the point is
an honest record of what evidence actually exists on disk, matching this
repo's audit/governance framing (see README — this is meant to read as
evidence for a reviewer, not just a notebook's own claim about itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class Artifact:
    label: str
    path: str
    description: str


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def artifact_trail(artifacts: list[Artifact]) -> pd.DataFrame:
    """Build a table of artifacts, checking existence/size/mtime live.

    A path a scenario *expected* to write but that isn't actually there
    shows up as a visible "missing" row, not a silent gap — if this ever
    disagrees with what the notebook just did, that's worth noticing.
    """
    rows = []
    for a in artifacts:
        p = Path(a.path)
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            exists = len(files) > 0
            size = _human_size(sum(f.stat().st_size for f in files)) if files else "—"
            modified = (
                datetime.fromtimestamp(max(f.stat().st_mtime for f in files)).strftime("%Y-%m-%d %H:%M:%S")
                if files
                else "—"
            )
        elif p.exists():
            stat = p.stat()
            exists = True
            size = _human_size(stat.st_size)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        else:
            exists = False
            size = "—"
            modified = "—"

        rows.append({
            "artifact": a.label,
            "path": a.path,
            "status": "present" if exists else "MISSING",
            "size": size,
            "last modified": modified,
            "description": a.description,
        })
    return pd.DataFrame(rows)
