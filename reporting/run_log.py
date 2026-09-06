"""Keep every run, and make it visible when a finding moves.

Two problems, one mechanism.

**Runs overwrite each other.** Each scenario writes fixed filenames into
`outputs/runs/<scenario>/`, so re-running replaces what came before. The
previous run is not archived, superseded, or diffed — it is gone.

**Nothing notices when a number changes.** Two real instabilities in this repo
were found by accident rather than by looking:

- Multi-Agent Handoff's `small_relay` went from 4/6 cases failing (*p* = 0.061,
  not significant) to 6/6 (*p* = 0.002, significant) between two runs of
  identical code against an identical fixture. The verdict in the doc changed.
- Resource & Budget's unbudgeted floor varied threefold on the same ticket with
  the same input.

Both are ordinary run-to-run variance. Neither is a defect. But a scenario
library whose published verdicts can flip without anyone seeing it has a
reporting problem, and the fix is cheap: keep the runs, and log one row each.

## What this is not

Not observability, not a hosted platform, and not a second copy of the raw data
in version control. `outputs/` is gitignored, so archives stay on the machine
that produced them. The flat files every notebook reads are still written
exactly as before — the archive is a copy taken alongside them, so nothing that
currently reads `outputs/runs/<scenario>/raw_results.csv` has to change.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

#: One ledger for the whole repo, beside the per-scenario run directories.
INDEX_PATH = Path("outputs/runs/index.csv")

#: Archived copies live under each scenario's own directory, so a scenario's
#: history travels with its artifacts rather than in a separate tree.
ARCHIVE_DIRNAME = "_archive"

INDEX_FIELDS = [
    "run_id", "scenario", "git_sha", "git_dirty", "model",
    "n_runs", "headline", "value", "notes", "archive",
]


def new_run_id() -> str:
    """UTC, second resolution, filesystem-safe and sorts chronologically."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def _repo_outputs_root() -> Path:
    return Path("outputs/runs").resolve()


def archive_run(scenario: str, output_dir: str | Path, *,
                headline: str = "", value: str = "", model: str = "",
                n_runs: int | str = "", notes: str = "") -> str | None:
    """Snapshot `output_dir`'s current files and append one row to the ledger.

    Returns the run id, or None when nothing was archived.

    Skips silently when `output_dir` is outside the repo's `outputs/runs/`.
    That is deliberate: regenerating a report from saved data — which several
    maintenance scripts do against a temporary directory — is not a new run,
    and logging it would put rows in the ledger that never touched a model.

    `value` is a string on purpose. A headline can be a rate, a ratio, or
    `4/6`; forcing it into a float would lose the distinction between "0.67"
    and "4 of 6 cases", and the second is what actually flipped a verdict here.
    """
    out = Path(output_dir)
    try:
        inside = _repo_outputs_root() in out.resolve().parents or out.resolve() == _repo_outputs_root()
    except OSError:
        inside = False
    if not inside:
        return None

    files = [f for f in out.glob("*") if f.is_file()]
    if not files:
        return None

    run_id = new_run_id()
    dest = out / ARCHIVE_DIRNAME / run_id
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest / f.name)

    row = {
        "run_id": run_id,
        "scenario": scenario,
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        # A run produced by uncommitted code is not reproducible from the SHA
        # alone, and that is worth knowing when a number moves.
        "git_dirty": "yes" if _git("status", "--porcelain") else "no",
        "model": model,
        "n_runs": n_runs,
        "headline": headline,
        "value": value,
        "notes": notes,
        "archive": str(dest),
    }
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not INDEX_PATH.exists()
    with INDEX_PATH.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)
    return run_id


def load_index():
    """The ledger as a DataFrame, or an empty one if no run has been logged."""
    import pandas as pd
    if not INDEX_PATH.exists():
        return pd.DataFrame(columns=INDEX_FIELDS)
    return pd.read_csv(INDEX_PATH)


def compare_runs(scenario: str | None = None):
    """Per scenario and headline, has the value moved between runs?

    The question the ledger exists to answer. `changed` is True when a headline
    took more than one distinct value across runs — which is the signal that a
    published verdict may rest on which run it happened to be written from.
    """
    import pandas as pd
    idx = load_index()
    if not len(idx):
        return idx
    if scenario:
        idx = idx[idx["scenario"] == scenario]
    rows = []
    for (sc, head), g in idx.groupby(["scenario", "headline"], dropna=False):
        g = g.sort_values("run_id")
        vals = [str(v) for v in g["value"]]
        rows.append({
            "scenario": sc,
            "headline": head,
            "runs": len(g),
            "first": vals[0],
            "latest": vals[-1],
            "distinct_values": len(set(vals)),
            "changed": len(set(vals)) > 1,
            "all_values": " → ".join(vals),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["changed", "scenario"], ascending=[False, True]).reset_index(drop=True)
