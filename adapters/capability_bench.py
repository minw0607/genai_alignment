"""Adapter onto genai_capability_bench.

Calls its stable public API (`run_from_config`) only — evaluators, metrics,
and client plumbing all stay in genai_capability_bench. This module just
resolves paths and loads the results back as DataFrames for genai_alignment's
own scenario notebooks and reporting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from genai_capability_bench.core.runner import run_from_config


def run_capability_scenario(config_path: str | Path) -> dict[str, object]:
    """Run a genai_capability_bench YAML config and load its results back.

    Parameters
    ----------
    config_path : str | Path
        Path to a genai_capability_bench-style config (models, dataset,
        pass_thresholds) — see configs/ for genai_alignment's own configs.

    Returns
    -------
    dict with "results" (per-task DataFrame), "summary" (aggregated
    DataFrame), and "output_dir" (Path to the run's artifacts).
    """
    output_dir = run_from_config(config_path)
    return {
        "results": pd.read_csv(output_dir / "results.csv"),
        "summary": pd.read_csv(output_dir / "summary.csv"),
        "output_dir": output_dir,
    }
