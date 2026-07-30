"""Repeat-run harness — run the same scenario N times and measure variance.

Shared across any scenario that needs "run this again and compare": first
used by consistency & reliability, and built with drift detection in mind —
docs/drift_detection.md's baseline-capture step needs the same underlying
mechanic (run the golden set N times, then compare a later run against what
was captured here).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def repeat_runs(run_fn: Callable[[], pd.DataFrame], n: int) -> list[pd.DataFrame]:
    """Call `run_fn()` n times, returning each call's results DataFrame."""
    return [run_fn() for _ in range(n)]


def combine_repeats(runs: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack N repeated runs into one long DataFrame, tagged with run_index."""
    return pd.concat(
        [r.assign(run_index=i) for i, r in enumerate(runs)],
        ignore_index=True,
    )


def pairwise_self_consistency(texts: list[str]) -> float:
    """Average pairwise TF-IDF cosine similarity among a set of texts.

    This measures whether N answers to the SAME question agree with each
    other — self-consistency, not accuracy. Deliberately not exact-match:
    two differently-worded but equivalent answers should score high here,
    which is why this is TF-IDF cosine similarity rather than string equality.
    Returns 1.0 if there's nothing to compare (0 or 1 texts), 0.0 if fewer
    than two are non-empty (can't compute a meaningful similarity).
    """
    if len(texts) <= 1:
        return 1.0
    non_empty = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(non_empty) < 2:
        return 0.0

    # Default token_pattern requires 2+ word characters, so single-character
    # answers (a bare "5", "No") tokenize to nothing and TfidfVectorizer raises
    # "empty vocabulary" — hence the looser pattern. If every text is still
    # unrepresentable (e.g. pure punctuation), fall back to exact-match rate
    # rather than letting a short-answer task crash the whole scenario.
    try:
        vectors = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(non_empty)
    except ValueError:
        normalized = [t.strip().lower() for t in non_empty]
        return float(len(set(normalized)) == 1)
    sim = cosine_similarity(vectors)
    n = len(non_empty)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return float(sum(sim[i, j] for i, j in pairs) / len(pairs))


def variance_by_task(
    combined: pd.DataFrame,
    id_col: str | list[str],
    score_col: str,
    passed_col: str,
    text_col: str | None = None,
    extra_numeric_cols: list[str] | None = None,
    passthrough_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Per-task variance summary across repeated runs.

    Deliberately configurable by column name rather than hardcoded to one
    result schema — the chatbot track's results (score/passed/actual_output)
    and the agentic track's (task_score/task_passed/tool_f1, no free-text
    answer to compare) both go through this same function with different
    column names, rather than two near-duplicate implementations.

    - `avg_<score_col>` / `<score_col>_std`: standard descriptive stats.
    - `pass_rate`: share of runs that passed; 0.0 or 1.0 means it never flipped.
    - `flips`: True if pass/fail was not the same in every run.
    - `self_consistency` (only if `text_col` given): average pairwise
      similarity among a task's N raw answers — catches "always passes, but
      says something different each time," which pass_rate alone would miss.
      Not meaningful for the agentic track (no comparable free-text answer),
      so `extra_numeric_cols` (e.g. tool_f1) plays that role there instead —
      variance in tool_f1 is "did it act consistently," the agentic analogue.
    """
    agg = {
        "n_runs": (score_col, "count"),
        f"avg_{score_col}": (score_col, "mean"),
        f"{score_col}_std": (score_col, "std"),
        "pass_rate": (passed_col, "mean"),
    }
    for col in extra_numeric_cols or []:
        agg[f"avg_{col}"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")

    grouped = combined.groupby(id_col).agg(**agg).reset_index()
    std_cols = [c for c in grouped.columns if c.endswith("_std")]
    grouped[std_cols] = grouped[std_cols].fillna(0.0)
    grouped["flips"] = ~grouped["pass_rate"].isin([0.0, 1.0])

    if text_col:
        self_consistency = (
            combined.groupby(id_col)[text_col]
            .apply(lambda s: pairwise_self_consistency(list(s)))
            .rename("self_consistency")
        )
        grouped = grouped.merge(self_consistency, on=id_col)

    for col in passthrough_cols or []:
        if col in combined.columns:
            values = combined.groupby(id_col)[col].first().rename(col)
            grouped = grouped.merge(values, on=id_col)

    return grouped
