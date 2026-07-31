"""Repeat-run harness — run the same scenario N times and measure variance.

Shared across any scenario that needs "run this again and compare": first
used by consistency & reliability, and built with drift detection in mind —
docs/drift_detection.md's baseline-capture step needs the same underlying
mechanic (run the golden set N times, then compare a later run against what
was captured here).

Statistical methodology, and why each piece replaced something weaker:

- **Semantic consistency** (`bidirectional_entailment`, `semantic_consistency`)
  clusters N sampled answers by whether each pair mutually entails the other,
  then reports 1 minus the normalized Shannon entropy over cluster sizes —
  this is the "semantic entropy" method (Kuhn et al., 2024, *Nature*,
  "Detecting hallucinations in large language models using semantic
  entropy"), adapted here to use an LLM-judge entailment check rather than a
  dedicated NLI model, so it reuses this repo's existing target-model
  infrastructure instead of adding a new ML dependency. This replaces
  `pairwise_self_consistency`'s TF-IDF cosine similarity against an
  unvalidated threshold — TF-IDF measures vocabulary overlap, not meaning,
  so two correct-but-differently-worded answers could score low under it.
  `pairwise_self_consistency` is kept below as a free, zero-API-call
  fallback, but is no longer what this repo's scenarios call by default.

- **Wilson score intervals** (`wilson_interval`, `add_wilson_ci`) replace a
  naive normal-approximation standard error for a pass-rate estimated from a
  small number of repeats (Wilson, 1927) — the normal approximation is
  poorly calibrated at n=5-10, which is exactly this repo's regime.

- **Benjamini-Hochberg FDR correction** (`benjamini_hochberg`,
  `add_reliability_significance`) corrects for testing many tasks
  simultaneously (Benjamini & Hochberg, 1995) — flagging a task's observed
  pass rate as "significantly" below an acceptable floor without this
  correction overstates how many findings are real, once you're running
  the same test across dozens of tasks at once.

See docs/consistency_reliability.md for the full write-up and references.
"""

from __future__ import annotations

import math
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

    Superseded by `semantic_consistency` below for anything that matters —
    this measures vocabulary overlap, not meaning, so it under-scores correct
    answers that are phrased differently each time. Kept as a free,
    zero-API-call fallback for contexts where spending LLM calls on an
    entailment check isn't worth it.
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


# ---------------------------------------------------------------- Semantic consistency
# Kuhn et al., 2024, Nature — "Detecting hallucinations in large language
# models using semantic entropy": cluster samples by bidirectional entailment,
# then measure entropy over the cluster-size distribution.

def bidirectional_entailment(client, text_a: str, text_b: str) -> bool:
    """True if text_a and text_b convey the same substantive meaning — each
    implies the other, checked via LLM judge rather than a dedicated NLI
    model (e.g. DeBERTa-MNLI, as the original paper uses), so this reuses the
    target model already configured for this repo instead of adding a new
    ML dependency. Exact-string matches short-circuit before spending a call.
    """
    if text_a.strip().lower() == text_b.strip().lower():
        return True
    prompt = (
        "Do these two answers convey the same substantive meaning, such that each "
        "implies the other (bidirectional entailment)? Answer with only one word: "
        "'yes' or 'no'.\n\n"
        f"Answer A: {text_a}\n\nAnswer B: {text_b}"
    )
    response = client.generate(prompt).text.strip().lower()
    return response.startswith("yes")


def semantic_consistency(texts: list[str], client) -> tuple[float, int]:
    """Cluster N texts by bidirectional entailment, return (consistency, n_clusters).

    Greedy clustering: each text joins the first existing cluster whose
    representative it mutually entails, or starts a new cluster — at most
    N-1 entailment checks in the worst case, not the full N-choose-2 pairs,
    and none at all for texts that are already exact-string duplicates.

    consistency = 1 - (Shannon entropy over cluster sizes / max possible
    entropy) — 1.0 means every answer landed in one cluster (fully
    consistent); 0.0 means every answer formed its own singleton cluster
    (every one meant something different). `n_clusters` — how many distinct
    *meanings* appeared across the N runs — is itself a useful diagnostic
    number independent of the normalized score.

    Missing/empty outputs are normalized to a sentinel string so repeated
    empty completions cluster together as "consistently produced no answer,"
    rather than being silently dropped or crashing the entailment check.
    """
    n = len(texts)
    if n <= 1:
        return 1.0, 1

    normalized = [t.strip() if isinstance(t, str) and t.strip() else "[NO OUTPUT]" for t in texts]

    clusters: list[list[str]] = []
    for text in normalized:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if text.lower() == rep.lower() or bidirectional_entailment(client, rep, text):
                cluster.append(text)
                placed = True
                break
        if not placed:
            clusters.append([text])

    sizes = [len(c) for c in clusters]
    probs = [s / n for s in sizes]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(n)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return 1.0 - normalized_entropy, len(clusters)


def add_semantic_consistency(
    variance: pd.DataFrame,
    combined: pd.DataFrame,
    id_col: str | list[str],
    text_col: str,
    client,
) -> pd.DataFrame:
    """Attach semantic_consistency + n_meaning_clusters columns to a variance table."""
    grouped = combined.groupby(id_col)[text_col].apply(lambda s: semantic_consistency(list(s), client))
    expanded = pd.DataFrame(
        grouped.tolist(), index=grouped.index, columns=["semantic_consistency", "n_meaning_clusters"]
    ).reset_index()
    return variance.merge(expanded, on=id_col)


def add_pairwise_consistency(
    variance: pd.DataFrame,
    combined: pd.DataFrame,
    id_col: str | list[str],
    text_col: str,
) -> pd.DataFrame:
    """Zero-API-call alternative to add_semantic_consistency, using TF-IDF
    similarity instead of bidirectional-entailment clustering — for dev
    iteration or cost-constrained runs where spending LLM calls on every
    task's consistency check isn't worth it. Same two output columns, so
    downstream code doesn't need to know which method produced them, but
    `n_meaning_clusters` here counts distinct normalized strings, not true
    meaning-clusters (TF-IDF has no notion of "same meaning, different
    words") — it will over-count clusters for paraphrased-but-equivalent
    answers, which is exactly the failure mode semantic_consistency exists
    to fix. Use this when speed/cost matters more than that distinction.
    """
    def _score_and_count(texts: list[str]) -> tuple[float, int]:
        score = pairwise_self_consistency(list(texts))
        distinct = len({t.strip().lower() for t in texts if isinstance(t, str) and t.strip()}) or 1
        return score, distinct

    grouped = combined.groupby(id_col)[text_col].apply(lambda s: _score_and_count(s))
    expanded = pd.DataFrame(
        grouped.tolist(), index=grouped.index, columns=["semantic_consistency", "n_meaning_clusters"]
    ).reset_index()
    return variance.merge(expanded, on=id_col)


# ---------------------------------------------------------------- Wilson confidence intervals
# Wilson, E. B. (1927), "Probable Inference, the Law of Succession, and
# Statistical Inference" — well-calibrated at small n, unlike the normal
# approximation (which can produce nonsensical intervals like [-0.1, 1.05]).

_Z_SCORES = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = _Z_SCORES.get(confidence, 1.96)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def add_wilson_ci(variance: pd.DataFrame, n_col: str = "n_runs", confidence: float = 0.95) -> pd.DataFrame:
    """Attach pass_rate_ci_low/high columns using the Wilson score interval."""
    result = variance.copy()
    successes = (result["pass_rate"] * result[n_col]).round().astype(int)
    intervals = [wilson_interval(int(s), int(n), confidence) for s, n in zip(successes, result[n_col])]
    result["pass_rate_ci_low"] = [i[0] for i in intervals]
    result["pass_rate_ci_high"] = [i[1] for i in intervals]
    return result


# ---------------------------------------------------------------- Multiple-comparison correction
# Benjamini, Y. & Hochberg, Y. (1995), "Controlling the False Discovery Rate:
# A Practical and Powerful Approach to Multiple Testing."

def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed directly (n is always small here)."""
    return sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(0, k + 1))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Which p-values remain significant after BH FDR correction, in original order."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    max_rank = -1
    for rank, idx in enumerate(order):
        threshold = (rank + 1) / n * alpha
        if p_values[idx] <= threshold:
            max_rank = rank
    significant = [False] * n
    for rank in range(max_rank + 1):
        significant[order[rank]] = True
    return significant


def add_reliability_significance(
    variance: pd.DataFrame,
    min_acceptable_rate: float,
    alpha: float = 0.05,
    n_col: str = "n_runs",
) -> pd.DataFrame:
    """Flag tasks whose pass rate is significantly below `min_acceptable_rate`,
    corrected for testing every task in this table at once.

    One-sided exact binomial test per task — H0: true pass rate >=
    min_acceptable_rate — then BH correction across all rows. This is the
    corrected version of `flips` (in `variance_by_task`): `flips` is a purely
    descriptive "did it ever disagree" flag that treats a 4-of-5 borderline
    case the same as a 1-of-5 wildly unstable one, and doesn't account for
    the fact that testing many tasks at once inflates how many will look
    "significant" by chance. `min_acceptable_rate` is a stated policy choice
    (see the scenario module for the value used and why), not something this
    function infers.
    """
    result = variance.copy()
    successes = (result["pass_rate"] * result[n_col]).round().astype(int)
    p_values = [
        _binom_cdf(int(s), int(n), min_acceptable_rate) for s, n in zip(successes, result[n_col])
    ]
    result["reliability_p_value"] = p_values
    result["significant_below_floor"] = benjamini_hochberg(p_values, alpha=alpha)
    return result


def variance_by_task(
    combined: pd.DataFrame,
    id_col: str | list[str],
    score_col: str,
    passed_col: str,
    extra_numeric_cols: list[str] | None = None,
    passthrough_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Per-task descriptive variance summary across repeated runs.

    Deliberately configurable by column name rather than hardcoded to one
    result schema — the chatbot track's results (score/passed) and the
    agentic track's (task_score/task_passed/tool_f1) both go through this
    same function with different column names, rather than two
    near-duplicate implementations.

    - `avg_<score_col>` / `<score_col>_std`: standard descriptive stats.
    - `pass_rate`: share of runs that passed; 0.0 or 1.0 means it never flipped.
    - `flips`: True if pass/fail was not the same in every run — a raw,
      uncorrected observation. See `add_reliability_significance` for the
      statistically corrected version of this same question, and
      `add_semantic_consistency` / `add_wilson_ci` for the other two
      refinements meant to be layered on top of this base table.
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

    for col in passthrough_cols or []:
        if col in combined.columns:
            values = combined.groupby(id_col)[col].first().rename(col)
            grouped = grouped.merge(values, on=id_col)

    return grouped
