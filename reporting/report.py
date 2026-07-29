"""Combine multi-dataset scenario runs and add LLM-judge second opinions.

Both helpers are generic across scenarios — any scenario that blends a public
benchmark with a custom golden set, or wants a judge pass on borderline scores,
can reuse these rather than re-deriving the pattern per notebook.
"""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd
from genai_capability_bench.clients.base import ModelClient
from genai_capability_bench.metrics.llm_judge import JudgeScore, judge_with_rubric


def _parse_metadata(value: Any) -> dict:
    """Metadata round-trips through CSV as a Python-literal string, not JSON."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def combine_runs(named_runs: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Combine several `run_capability_scenario` outputs into one report.

    Parameters
    ----------
    named_runs : dict[str, dict]
        Maps a human-readable sub-dataset label (e.g. "custom_golden_set") to
        the dict returned by `adapters.capability_bench.run_capability_scenario`.

    Returns
    -------
    dict with "results" (all rows, tagged with a `dataset_label` column) and
    "summary" (aggregated by dataset_label + category).
    """
    frames = []
    for label, out in named_runs.items():
        df = out["results"].copy()
        df["dataset_label"] = label
        df["dataset_key"] = df["metadata"].apply(lambda m: _parse_metadata(m).get("dataset_key"))
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)
    summary = (
        results.groupby(["dataset_label", "category"], dropna=False)
        .agg(n=("task_id", "count"), avg_score=("score", "mean"), pass_rate=("passed", "mean"))
        .reset_index()
    )
    summary["avg_score"] = summary["avg_score"].round(4)
    summary["pass_rate"] = summary["pass_rate"].round(4)
    return {"results": results, "summary": summary}


def judge_borderline(
    results: pd.DataFrame,
    client: ModelClient,
    rubric: str,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Add an LLM-judge second opinion to every row scoring below `threshold`.

    Reuses genai_capability_bench's `judge_with_rubric` — no judge logic is
    reimplemented here. Rows at or above threshold are left unjudged (judge
    columns are NA) since they already passed on the deterministic metric.

    Caveat: if `client` is the same model family as the target being tested,
    this is not an independent check — note that in any report that uses it.
    """
    out = results.copy()
    out["judge_score"] = pd.NA
    out["judge_reason"] = pd.NA

    for idx, row in out.iterrows():
        if row["score"] >= threshold:
            continue
        result: JudgeScore = judge_with_rubric(
            client,
            task=row["input_text"],
            answer=row["actual_output"],
            rubric=rubric,
            reference=row.get("expected_output"),
        )
        out.at[idx, "judge_score"] = result.score
        out.at[idx, "judge_reason"] = result.reason

    return out
