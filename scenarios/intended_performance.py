"""Scenario-specific logic for Intended Performance (Tier 1).

Kept out of the notebook so the notebook stays a thin, readable narrative —
see notebooks/01_intended_performance.ipynb and docs/intended_performance.md.
Everything specific to this one scenario lives here: its two golden sets,
its charts, and how its report gets built. Generic cross-scenario pieces
(multi-dataset combination, judge review, the HTML report template) live in
reporting/, and are only called from here, not reimplemented.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from genai_capability_bench.clients.factory import create_client
from genai_capability_bench.core.schemas import ModelSpec

from adapters.capability_bench import run_capability_scenario
from reporting.artifacts import Artifact
from reporting.html_report import ChartImage, DataSection, Metric, ScenarioReport, fig_to_base64
from reporting.report import combine_runs, judge_borderline

PALETTE = {"pass": "#2a9d8f", "fail": "#e76f51", "custom": "#264653", "public": "#e9c46a"}

GOLDEN_SET_PATH = "scenarios/fixtures/intended_performance.jsonl"
PUBLIC_SAMPLE_PATH = "scenarios/fixtures/public_benchmark_sample.jsonl"
CUSTOM_CONFIG = "configs/intended_performance.yaml"
PUBLIC_CONFIG = "configs/intended_performance_public_benchmark.yaml"
PASS_THRESHOLD = 0.7

JUDGE_RUBRIC = (
    "Score 1.0 if the answer is substantively correct and consistent with the reference, "
    "regardless of phrasing or verbosity. Score 0.0 if it is factually wrong or contradicts "
    "the reference. Briefly say why."
)

REVIEW_COLS = ["task_id", "dataset_label", "expected_output", "actual_output", "score", "judge_score", "judge_reason"]


# ---------------------------------------------------------------- Data

def load_golden_set() -> pd.DataFrame:
    df = pd.read_json(GOLDEN_SET_PATH, lines=True)
    df["trap_type"] = df["metadata"].apply(lambda m: m.get("trap_type"))
    return df


def load_public_sample() -> pd.DataFrame:
    df = pd.read_json(PUBLIC_SAMPLE_PATH, lines=True)
    df["source_dataset"] = df["metadata"].apply(lambda m: m.get("source_dataset"))
    return df


def public_sample_composition(public_sample: pd.DataFrame) -> pd.DataFrame:
    return public_sample.groupby(["source_dataset", "category"]).size().reset_index(name="n")


def plot_data_structure(golden_set: pd.DataFrame, public_sample: pd.DataFrame) -> ChartImage:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    trap_counts = golden_set["trap_type"].value_counts()
    axes[0].barh(trap_counts.index[::-1], trap_counts.values[::-1], color=PALETTE["custom"])
    axes[0].set_title(f"Custom golden set\nby trap type (n={len(golden_set)})")
    axes[0].set_xlabel("tasks")

    source_counts = public_sample["source_dataset"].value_counts()
    axes[1].pie(
        source_counts.values, labels=source_counts.index, autopct="%1.0f%%",
        colors=["#e9c46a", "#f4a261", "#e76f51"], startangle=90,
    )
    axes[1].set_title(f"Public benchmark sample\nby source (n={len(public_sample)})")

    cat_counts = public_sample["category"].value_counts()
    axes[2].barh(cat_counts.index[::-1], cat_counts.values[::-1], color=PALETTE["public"])
    axes[2].set_title(f"Public benchmark sample\nby category (n={len(public_sample)})")
    axes[2].set_xlabel("tasks")

    plt.tight_layout()
    chart = ChartImage(
        title="Data structure at a glance",
        caption=(
            "Left: the custom golden set's trap-type coverage. Center/right: the public "
            "benchmark sample's source and category composition (stratified sample, seed 42)."
        ),
        base64_png=fig_to_base64(fig),
        section="data",
    )
    plt.show()
    return chart


# ---------------------------------------------------------------- Scoring

def metrics_table() -> pd.DataFrame:
    from genai_capability_bench.metrics.registry import METRIC_SPECS

    metrics_used = ["exact_match", "token_f1", "contains_match", "semantic_similarity", "llm_judge_correctness"]
    return pd.DataFrame([
        {
            "metric": METRIC_SPECS[m].display_name,
            "definition": METRIC_SPECS[m].definition,
            "best for": METRIC_SPECS[m].best_for,
            "limitation": METRIC_SPECS[m].limitations,
        }
        for m in metrics_used
    ])


# ---------------------------------------------------------------- Run

def run() -> tuple[dict, str, str]:
    """Run both sub-datasets through the adapter and combine. Returns (combined, target_model, api_version)."""
    target_model = os.environ.get("TARGET_MODEL", "<unset>")
    api_version = os.environ.get("OPENAI_API_VERSION", "<unset>")

    custom_run = run_capability_scenario(CUSTOM_CONFIG)
    public_run = run_capability_scenario(PUBLIC_CONFIG)
    combined = combine_runs({
        "custom_golden_set": custom_run,
        "public_benchmark_sample": public_run,
    })
    return combined, target_model, api_version


def target_summary_line(target_model: str, api_version: str) -> str:
    return (
        f"**Target:** Azure OpenAI · `{target_model}` · API version `{api_version}` · "
        "temperature omitted, `max_completion_tokens` used as the token parameter "
        "(required for reasoning-family models that reject a bare `temperature`/`max_tokens` request)."
    )


# ---------------------------------------------------------------- Results

def plot_score_by_task(results: pd.DataFrame) -> ChartImage:
    results = results.sort_values(["dataset_label", "task_id"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, (label, group) in zip(axes, results.groupby("dataset_label")):
        colors = group["passed"].map({True: PALETTE["pass"], False: PALETTE["fail"]})
        ax.bar(range(len(group)), group["score"], color=colors)
        ax.axhline(PASS_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels(
            group["task_id"].str.replace("answer_accuracy_knowledge_v1_", "", regex=False),
            rotation=75, ha="right", fontsize=7,
        )
        ax.set_title(label)
        ax.set_ylabel("score")
    fig.suptitle(f"Score by task (green = pass, red = below {PASS_THRESHOLD:.2f} threshold)")
    plt.tight_layout()
    chart = ChartImage(
        title="Score by task",
        caption="Every task in both sub-datasets, colored by pass/fail against the 0.70 threshold.",
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


def plot_pass_rate(results: pd.DataFrame) -> ChartImage:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (label, group) in zip(axes, results.groupby("dataset_label")):
        pass_n = int(group["passed"].sum())
        fail_n = len(group) - pass_n
        ax.pie(
            [pass_n, fail_n], labels=[f"pass ({pass_n})", f"below threshold ({fail_n})"],
            colors=[PALETTE["pass"], PALETTE["fail"]], autopct="%1.0f%%", startangle=90,
        )
        ax.set_title(label)
    fig.suptitle("Pass rate by sub-dataset")
    plt.tight_layout()
    chart = ChartImage(
        title="Pass rate by sub-dataset",
        caption="Share of tasks at or above the 0.70 deterministic-metric threshold, before judge review.",
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


# ---------------------------------------------------------------- Judge review

def judge_review(results: pd.DataFrame, target_model: str) -> pd.DataFrame:
    """`target_model` is only a fallback — the judge should be a genuinely
    different model from the one under test, so this reads JUDGE_MODEL from
    the environment first (see .env.example)."""
    judge_spec = ModelSpec(
        name="judge",
        provider="azure_openai",
        model=os.environ.get("JUDGE_MODEL", target_model),
        temperature=None,
        max_tokens=200,
        token_parameter="max_completion_tokens",
    )
    judge_client = create_client(judge_spec)
    judged = judge_borderline(results, judge_client, JUDGE_RUBRIC, threshold=PASS_THRESHOLD)
    judged_subset = judged[judged["judge_score"].notna()].copy()
    judged_subset["score"] = judged_subset["score"].round(3)
    return judged_subset


def plot_judge_disposition(judged_subset: pd.DataFrame) -> ChartImage | None:
    n_judged = len(judged_subset)
    if not n_judged:
        print("No sub-threshold tasks this run — nothing for the judge to review.")
        return None

    n_upgraded = int((judged_subset["judge_score"] >= PASS_THRESHOLD).sum())
    n_confirmed_wrong = n_judged - n_upgraded

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["scoring-metric\nartifact", "genuine\nmiss"], [n_upgraded, n_confirmed_wrong], color=["#f4a261", "#e76f51"])
    ax.set_ylabel("tasks")
    ax.set_title(f"Judge disposition of {n_judged} sub-threshold task(s)")
    for i, v in enumerate([n_upgraded, n_confirmed_wrong]):
        ax.text(i, v + 0.05, str(v), ha="center")
    plt.tight_layout()
    chart = ChartImage(
        title="Judge disposition of sub-threshold tasks",
        caption=(
            "Of the tasks that scored below 0.70 on the deterministic metric, how many the LLM "
            "judge assessed as actually correct (a scoring-metric artifact) vs. genuinely wrong."
        ),
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


# ---------------------------------------------------------------- Report

def _build_observations(judged_subset: pd.DataFrame) -> list[str]:
    """Built from this run's actual judge output, not hand-written, so it can't go stale
    on a re-run that produces a different mix of sub-threshold cases."""
    observations = []

    custom_judged = judged_subset[judged_subset["dataset_label"] == "custom_golden_set"]
    if len(custom_judged):
        custom_upgraded = int((custom_judged["judge_score"] >= PASS_THRESHOLD).sum())
        observations.append(
            f"Sub-threshold scores are not automatically task failures — {custom_upgraded} of "
            f"{len(custom_judged)} sub-threshold case(s) in the **custom golden set** this run were "
            "judged substantively correct, because `short_answer_qa` under-credits a correct answer "
            "phrased as a full sentence rather than matching a short reference phrasing closely. Always "
            "read the judge's reasoning before treating a low score as a real miss."
        )

    public_judged = judged_subset[judged_subset["dataset_label"] == "public_benchmark_sample"]
    if len(public_judged):
        public_wrong = int((public_judged["judge_score"] < PASS_THRESHOLD).sum())
        sand_flagged = "answer_accuracy_knowledge_v1_Mercury_7128923" in public_judged["task_id"].values
        obs = (
            f"The **public benchmark sample** told a different story: the judge confirmed {public_wrong} "
            f"of {len(public_judged)} sub-threshold case(s) as genuinely wrong, not scoring artifacts — "
            "a reminder that sub-threshold scores need to be checked, not assumed identical in cause "
            "across sub-datasets."
        )
        if sand_flagged:
            obs += (
                ' One case is worth a caveat of its own — "which substance retains the most energy '
                'from the Sun" (gold answer "sand") is scientifically debatable on '
                "specific-heat-capacity grounds, a reminder that a public benchmark's gold answer isn't "
                "automatically ground truth either."
            )
        observations.append(obs)

    empty_output_n = int(judged_subset["actual_output"].isna().sum())
    if empty_output_n:
        observations.append(
            f"{empty_output_n} sub-threshold case(s) this run had no visible model output at all — an "
            "empty completion, not an error. This has now been observed across multiple runs of this "
            "notebook; the 300-token `max_completion_tokens` budget may be tight enough for a "
            "reasoning-family model to exhaust it on internal reasoning before producing a visible "
            "answer, rather than this being a one-off."
        )

    observations.append(
        "Changing the target model changed the result: an earlier run of this exact scenario against a "
        "different model in the `gpt-5` family produced a materially different pass rate on the "
        "identical golden set. That's a live instance of Tier 2 — drift detection surfacing inside a "
        "Tier 1 test, and a concrete reason results here should be re-read after any model change."
    )
    return observations


def _build_next_steps(judged_subset: pd.DataFrame) -> list[str]:
    next_steps = [
        "Wire `llm_judge_correctness` in as a proper secondary metric in `genai_capability_bench`, "
        "rather than the ad hoc post-hoc pass used here.",
        "Route any case where the judge disagrees with a human reviewer for adjudication, rather "
        "than trusting the same-model-family judge as final.",
    ]
    empty_output_n = int(judged_subset["actual_output"].isna().sum()) if len(judged_subset) else 0
    if empty_output_n:
        next_steps.insert(
            0,
            "Increase `max_completion_tokens` for this scenario's config and confirm the "
            "empty-completion cases stop recurring.",
        )
    return next_steps


def build_report(
    golden_set: pd.DataFrame,
    public_sample: pd.DataFrame,
    combined: dict,
    judged_subset: pd.DataFrame,
    charts: list[ChartImage],
    target_model: str,
    api_version: str,
) -> ScenarioReport:
    overall_n = len(combined["results"])
    overall_pass_rate = combined["results"]["passed"].mean()
    overall_avg_score = combined["results"]["score"].mean()

    by_dataset = combined["summary"].groupby("dataset_label").apply(
        lambda g: pd.Series({
            "n": int(g["n"].sum()),
            "avg_score": round((g["avg_score"] * g["n"]).sum() / g["n"].sum(), 3),
            "pass_rate": round((g["pass_rate"] * g["n"]).sum() / g["n"].sum(), 3),
        })
    ).reset_index()
    by_dataset["n"] = by_dataset["n"].astype(int)

    n_judged = len(judged_subset)
    n_upgraded = int((judged_subset["judge_score"] >= PASS_THRESHOLD).sum()) if n_judged else 0
    n_confirmed_wrong = n_judged - n_upgraded

    return ScenarioReport(
        scenario_name="Intended Performance",
        tier="Tier 1",
        risk="Silent task failure or plausible-looking wrong output.",
        goal="Performs its defined task correctly and completely.",
        target_summary={
            "Provider": "Azure OpenAI",
            "Model": target_model,
            "API version": api_version,
            "Generation config": "temperature omitted; max_completion_tokens token parameter (reasoning-family model)",
        },
        approach=(
            "Adapter onto `genai_capability_bench`'s `AnswerAccuracyEvaluator` via its stable "
            "`run_from_config` API (no evaluator/metric code duplicated). Two sub-datasets run through "
            "the identical harness and are reported side by side. Every task scoring below the 0.70 pass "
            "threshold gets an LLM-judge second opinion (same-model-family caveat noted below) rather "
            "than being taken at face value."
        ),
        data_sections=[
            DataSection(
                name="Custom HR/IT policy golden set",
                layer="Layer 6 — custom-authored",
                source="Hand-written for this repo",
                size=f"{len(golden_set)} tasks",
                description=(
                    "Built to produce plausible-looking wrong answers via boundary values, negation, "
                    "overriding exceptions, and conditional rules. Entirely synthetic."
                ),
            ),
            DataSection(
                name="Public benchmark sample",
                layer="Layer 1 — generic benchmark",
                source="MMLU + TriviaQA + ARC, via genai_capability_bench's curated_knowledge_v1",
                size=f"{len(public_sample)} tasks (stratified from 33,156)",
                description="Reproducible, externally recognized benchmarks for breadth and comparability.",
            ),
        ],
        key_metrics=[
            Metric(value=f"{overall_avg_score:.2f}", label="Overall avg score", sublabel=f"n={overall_n}"),
            Metric(value=f"{overall_pass_rate:.0%}", label="Overall pass rate"),
            Metric(value=str(n_judged), label="Sub-threshold tasks reviewed by judge"),
            Metric(value=str(n_upgraded), label="Confirmed scoring-metric artifacts"),
            Metric(value=str(n_confirmed_wrong), label="Confirmed genuine misses"),
        ],
        results_tables=[
            ("Results by sub-dataset", by_dataset),
            ("Judge review of sub-threshold tasks", judged_subset[REVIEW_COLS] if n_judged else judged_subset),
        ],
        charts=charts,
        observations=_build_observations(judged_subset),
        next_steps=_build_next_steps(judged_subset),
        notebook_link="../notebooks/01_intended_performance.ipynb",
        doc_link="../docs/intended_performance.md",
    )


def artifacts(report_path: str) -> list[Artifact]:
    """Every file this scenario's run reads or produces, for the documentation trail."""
    return [
        Artifact(
            "Custom golden set (input)", GOLDEN_SET_PATH,
            "Hand-authored HR/IT policy Q&A tasks — versioned, not regenerated per run.",
        ),
        Artifact(
            "Public benchmark sample (input)", PUBLIC_SAMPLE_PATH,
            "Frozen stratified MMLU/TriviaQA/ARC sample (seed 42) — versioned, not regenerated per run.",
        ),
        Artifact(
            "Custom golden set — run output", "outputs/runs/intended_performance_v1",
            "Per-task results, summary, and run metadata from genai_capability_bench (gitignored, regenerated every run).",
        ),
        Artifact(
            "Public benchmark — run output", "outputs/runs/intended_performance_public_benchmark_v1",
            "Per-task results, summary, and run metadata from genai_capability_bench (gitignored, regenerated every run).",
        ),
        Artifact(
            "HTML testing report", report_path,
            "The rendered report embedded above — scope, data, results, charts, and judge review.",
        ),
    ]
