"""Scenario-specific logic for Consistency & Reliability (Tier 1).

Kept out of the notebook so the notebook stays a thin, readable narrative —
see notebooks/02_consistency_reliability.ipynb and docs/consistency_reliability.md.

Two tracks, testing two different kinds of "same input, run again":

- Chatbot track: the same two golden sets from scenario 1 (Intended
  Performance), each run N times through genai_capability_bench's evaluator.
  "Consistent" = pass/fail doesn't flip across runs, and the raw answers are
  self-consistent with each other (TF-IDF similarity, not exact match).
- Agentic track: a fixed Mind2Web task subset, run N times through
  multi_agent_otel_eval, in both single-agent and multi-agent mode.
  "Consistent" here is NOT text similarity — an agent's actions are compared
  via tool_f1 variance (already equivalence-aware upstream: calling a
  differently-named-but-equivalent tool isn't penalized) and task-outcome
  flip rate, not by diffing raw trajectories.

Generic repeat-run mechanics (combine_repeats, variance_by_task,
pairwise_self_consistency) live in reporting/repeat_run.py, not here, since
drift detection will need the same "run N times, compare" mechanic later.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from adapters.agent_otel import AgentHarness
from adapters.capability_bench import run_capability_scenario
from reporting.artifacts import Artifact
from reporting.html_report import ChartImage, DataSection, Metric, ScenarioReport, fig_to_base64
from reporting.repeat_run import combine_repeats, repeat_runs, variance_by_task

PALETTE = {"pass": "#2a9d8f", "fail": "#e76f51", "single": "#264653", "multi": "#e9c46a"}

CUSTOM_CONFIG = "configs/intended_performance.yaml"
PUBLIC_CONFIG = "configs/intended_performance_public_benchmark.yaml"
PASS_THRESHOLD = 0.7

CHATBOT_N_REPEATS = 5
AGENTIC_N_REPEATS = 5
AGENTIC_N_TASKS = 5

# genai_capability_bench's run_capability_scenario writes each repeat to the
# same fixed run_id directory (configs/intended_performance*.yaml), so calling
# it N times in a loop overwrites itself — only the last repeat would survive
# on disk. This scenario's own output dir is where all N repeats' combined
# raw results actually persist; see save_artifacts().
OUTPUT_DIR = "outputs/runs/consistency_reliability"


# ---------------------------------------------------------------- Chatbot track

def run_chatbot_repeats(n: int = CHATBOT_N_REPEATS) -> dict:
    """Run both scenario-1 golden sets n times each. Returns combined long-format
    results (one row per task per run) tagged with dataset_label."""
    frames = []
    for label, config_path in [
        ("custom_golden_set", CUSTOM_CONFIG),
        ("public_benchmark_sample", PUBLIC_CONFIG),
    ]:
        runs = repeat_runs(lambda cp=config_path: run_capability_scenario(cp)["results"], n)
        combined = combine_repeats(runs)
        combined["dataset_label"] = label
        frames.append(combined)
    return {"results": pd.concat(frames, ignore_index=True), "n_repeats": n}


def chatbot_variance(chatbot_results: pd.DataFrame) -> pd.DataFrame:
    return variance_by_task(
        chatbot_results,
        id_col="task_id",
        score_col="score",
        passed_col="passed",
        text_col="actual_output",
        passthrough_cols=["dataset_label", "category"],
    )


def plot_chatbot_variance(variance: pd.DataFrame) -> ChartImage:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    ordered = variance.sort_values(["dataset_label", "task_id"])
    colors = ordered["flips"].map({True: PALETTE["fail"], False: PALETTE["pass"]})
    axes[0].bar(range(len(ordered)), ordered["score_std"], color=colors)
    axes[0].set_xticks(range(len(ordered)))
    axes[0].set_xticklabels(
        ordered["task_id"].str.replace("answer_accuracy_knowledge_v1_", "", regex=False),
        rotation=75, ha="right", fontsize=7,
    )
    axes[0].set_ylabel("score std dev across runs")
    axes[0].set_title("Score variance by task (red = pass/fail flipped)")

    axes[1].bar(range(len(ordered)), ordered["self_consistency"], color=PALETTE["single"])
    axes[1].set_xticks(range(len(ordered)))
    axes[1].set_xticklabels(
        ordered["task_id"].str.replace("answer_accuracy_knowledge_v1_", "", regex=False),
        rotation=75, ha="right", fontsize=7,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("avg pairwise similarity")
    axes[1].set_title("Self-consistency of raw answers across runs")

    plt.tight_layout()
    chart = ChartImage(
        title="Chatbot repeat-run variance",
        caption=(
            "Left: how much each task's score moved across repeated runs (red bars flipped "
            "pass/fail at least once). Right: how similar the N raw answers were to each other "
            "(TF-IDF cosine similarity, not correctness) — a task can pass every run and still "
            "score low here if it phrases the answer differently each time."
        ),
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


# ---------------------------------------------------------------- Agentic track

def build_agent_harness(target_model: str) -> AgentHarness:
    return AgentHarness(target_model)


def run_agentic_repeats(
    harness: AgentHarness,
    n_tasks: int = AGENTIC_N_TASKS,
    n_repeats: int = AGENTIC_N_REPEATS,
) -> dict:
    """Run a fixed Mind2Web task subset n_repeats times, in both single- and
    multi-agent mode. Returns combined long-format results tagged with mode."""
    tasks = harness.load_tasks(n_tasks)
    frames = []
    for mode in ["single", "multi"]:
        runs = repeat_runs(lambda t=tasks, m=mode: harness.run(t, mode=m), n_repeats)
        combined = combine_repeats(runs)
        combined["mode"] = mode
        frames.append(combined)
    return {
        "results": pd.concat(frames, ignore_index=True),
        "n_tasks": n_tasks,
        "n_repeats": n_repeats,
        "tasks": tasks,
    }


def agentic_variance(agentic_results: pd.DataFrame) -> pd.DataFrame:
    # Mind2Web task_ids are 0..n-1 *within each mode's own run*, not globally
    # unique — grouping by task_id alone would silently merge single- and
    # multi-agent results for "task 0" into one row. Group by the pair instead.
    return variance_by_task(
        agentic_results,
        id_col=["task_id", "mode"],
        score_col="task_score",
        passed_col="task_passed",
        extra_numeric_cols=["tool_f1", "n_tool_calls"],
        passthrough_cols=["website"],
    )


def plot_agentic_variance(variance: pd.DataFrame) -> ChartImage:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (mode, group) in zip(axes, variance.groupby("mode")):
        colors = group["flips"].map({True: PALETTE["fail"], False: PALETTE["pass"]})
        ax.bar(group["task_id"].astype(str), group["task_score_std"], color=colors)
        ax.set_xlabel("task_id")
        ax.set_ylabel("task_score std dev across runs")
        ax.set_title(f"{mode}-agent: score variance (red = outcome flipped)")

    plt.tight_layout()
    chart = ChartImage(
        title="Agentic repeat-run variance",
        caption=(
            "Task-score variance across repeated runs of the identical Mind2Web tasks, "
            "single-agent vs. multi-agent. Red bars flipped pass/fail at least once across runs."
        ),
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


def plot_tool_consistency(variance: pd.DataFrame) -> ChartImage:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, group in variance.groupby("mode"):
        ax.scatter(group["avg_tool_f1"], group["tool_f1_std"], label=mode, s=60)
    ax.set_xlabel("avg tool_f1 across runs")
    ax.set_ylabel("tool_f1 std dev across runs")
    ax.set_title("Action consistency: tool-selection variance per task")
    ax.legend()
    plt.tight_layout()
    chart = ChartImage(
        title="Action consistency across repeated runs",
        caption=(
            "tool_f1 is already equivalence-aware upstream (multi_agent_otel_eval's flexible "
            "tool-equivalence mapping) — high variance here means the agent's tool choices "
            "genuinely differed run to run, not just that it phrased things differently."
        ),
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


# ---------------------------------------------------------------- Report

def _chatbot_observations(chatbot_var: pd.DataFrame) -> list[str]:
    obs = []
    n_flips = int(chatbot_var["flips"].sum())
    obs.append(
        f"{n_flips} of {len(chatbot_var)} tasks flipped pass/fail across "
        f"{CHATBOT_N_REPEATS} repeated runs of the identical prompt — with temperature "
        "omitted for this reasoning-family model, meaning this variance isn't coming from "
        "an explicit sampling knob we could just turn down."
    )
    low_consistency = chatbot_var[chatbot_var["self_consistency"] < 0.5]
    if len(low_consistency):
        obs.append(
            f"{len(low_consistency)} task(s) had low self-consistency (< 0.5 average pairwise "
            "similarity) despite mostly passing — the model answers correctly every time but "
            "phrases it differently enough that a naive text-diff would flag it as unstable. "
            "Pass-rate alone would have missed this; that's why self-consistency is tracked "
            "as its own signal, not folded into pass/fail."
        )
    return obs


def _agentic_observations(agentic_var: pd.DataFrame) -> list[str]:
    obs = []
    tool_f1_flat = agentic_var["tool_f1_std"].max() < 0.01
    if tool_f1_flat:
        obs.append(
            "`tool_f1` showed essentially zero variance across every task and mode this run, "
            f"even though `n_tool_calls` did not (std up to {agentic_var['n_tool_calls_std'].max():.0f} "
            "calls on the same task) — Tool Correctness, as this framework scores it, appears to be "
            "driven by whether the required actions were performed at least once, not by how much "
            "extra exploration or redundant tool-calling happened around them. That means `tool_f1_std` "
            "isn't the sensitive signal for agentic action-consistency this scenario expected it to be; "
            "`n_tool_calls_std` is carrying more of the real behavioral-variance signal in this run."
        )
    for mode, group in agentic_var.groupby("mode"):
        n_flips = int(group["flips"].sum())
        detail = (
            f"average tool_f1 std dev {group['tool_f1_std'].mean():.2f}"
            if not tool_f1_flat
            else f"average n_tool_calls std dev {group['n_tool_calls_std'].mean():.1f}"
        )
        obs.append(
            f"{mode}-agent: {n_flips} of {len(group)} tasks flipped pass/fail across "
            f"{AGENTIC_N_REPEATS} repeated runs; {detail}."
        )
    return obs


def build_report(
    chatbot_var: pd.DataFrame,
    agentic_var: pd.DataFrame,
    target_model: str,
    api_version: str,
    charts: list[ChartImage],
) -> ScenarioReport:
    overall_flip_rate_chat = chatbot_var["flips"].mean()
    overall_flip_rate_agent = agentic_var["flips"].mean()

    observations = _chatbot_observations(chatbot_var) + _agentic_observations(agentic_var)
    observations.append(
        "The chatbot track (a single LLM call per task) and the agentic track (a multi-step "
        "orchestration loop, each step re-invoking the model) aren't a controlled ablation of "
        "'model vs. harness' variance — that would need the same harness run with orchestration "
        "on and off, which neither sibling repo supports today. What this comparison *does* show "
        "honestly: the chatbot track is close to a variance floor (one call, no orchestration); "
        "the agentic track's variance sits on top of that floor, compounded across every "
        "intermediate step the orchestration takes. Treat the agentic numbers as an upper bound "
        "on total variance, not an isolated 'harness contribution.'"
    )

    next_steps = [
        "Build a controlled ablation (same task, orchestration disabled vs. enabled) to actually "
        "isolate how much variance the harness adds on top of the model's own — the honest gap "
        "flagged in the observations above.",
        "Extend self-consistency scoring to semantic (embedding-based) similarity instead of "
        "TF-IDF, which will under-count paraphrases that use very different vocabulary.",
    ]

    return ScenarioReport(
        scenario_name="Consistency & Reliability",
        tier="Tier 1",
        risk="Same input yields different output; hallucination / variance.",
        goal="Repeatable, consistent outputs for equivalent inputs.",
        target_summary={
            "Provider": "Azure OpenAI",
            "Model": target_model,
            "API version": api_version,
            "Chatbot track": f"{CHATBOT_N_REPEATS} repeats of scenario 1's 40-task golden set",
            "Agentic track": (
                f"{AGENTIC_N_REPEATS} repeats of {AGENTIC_N_TASKS} Mind2Web tasks, "
                "single-agent and multi-agent mode"
            ),
        },
        approach=(
            "Two tracks, testing two different kinds of 'same input, run again.' Chatbot: reuses "
            "scenario 1's two golden sets and adapter unchanged, run N times via "
            "`reporting/repeat_run.py`'s generic repeat harness. Agentic: a new adapter "
            "(`adapters/agent_otel.py`) onto `multi_agent_otel_eval`'s `evaluate_batch()`, run N "
            "times on a fixed Mind2Web task subset in both single- and multi-agent mode. Both "
            "tracks share the same variance-scoring function "
            "(`reporting.repeat_run.variance_by_task`), configured with different column names "
            "rather than duplicated — the chatbot track adds self-consistency of the raw text "
            "answers; the agentic track adds tool_f1 variance instead, since comparing raw agent "
            "trajectories as text wouldn't be the right notion of 'consistent action.'"
        ),
        data_sections=[
            DataSection(
                name="Chatbot: scenario 1's two golden sets",
                layer="Reused, not new",
                source="Custom HR/IT policy set + public MMLU/TriviaQA/ARC sample",
                size=f"40 tasks × {CHATBOT_N_REPEATS} repeats = {40 * CHATBOT_N_REPEATS} calls",
                description="Identical data to scenario 1 — this scenario only adds the repeat dimension.",
            ),
            DataSection(
                name="Agentic: Mind2Web task subset",
                layer="New for this scenario",
                source="multi_agent_otel_eval's Mind2Web loader (NeurIPS 2023 web-navigation benchmark)",
                size=(
                    f"{AGENTIC_N_TASKS} tasks × {AGENTIC_N_REPEATS} repeats × 2 modes = "
                    f"{AGENTIC_N_TASKS * AGENTIC_N_REPEATS * 2} agent runs"
                ),
                description="Same fixed task subset re-run identically in single-agent and multi-agent mode.",
            ),
        ],
        key_metrics=[
            Metric(value=f"{overall_flip_rate_chat:.0%}", label="Chatbot tasks that flipped pass/fail"),
            Metric(value=f"{overall_flip_rate_agent:.0%}", label="Agentic tasks that flipped pass/fail"),
            Metric(value=f"{chatbot_var['self_consistency'].mean():.2f}", label="Avg chatbot self-consistency"),
            Metric(value=f"{agentic_var['tool_f1_std'].mean():.2f}", label="Avg agentic tool_f1 std dev"),
        ],
        results_tables=[
            ("Chatbot variance by task", chatbot_var),
            ("Agentic variance by task and mode", agentic_var),
        ],
        charts=charts,
        observations=observations,
        next_steps=next_steps,
        notebook_link="../notebooks/02_consistency_reliability.ipynb",
        doc_link="../docs/consistency_reliability.md",
    )


def save_artifacts(
    chatbot_results: pd.DataFrame,
    chatbot_var: pd.DataFrame,
    agentic_results: pd.DataFrame,
    agentic_var: pd.DataFrame,
) -> dict[str, str]:
    """Persist the combined raw results and variance tables for both tracks.

    Without this, only the last of the N repeats would survive on disk (see
    the OUTPUT_DIR note above) — the aggregated variance numbers in the
    report would have no underlying per-repeat evidence to audit against.
    """
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "chatbot_raw": out_dir / "chatbot_raw_results.csv",
        "chatbot_variance": out_dir / "chatbot_variance.csv",
        "agentic_raw": out_dir / "agentic_raw_results.csv",
        "agentic_variance": out_dir / "agentic_variance.csv",
    }
    chatbot_results.to_csv(paths["chatbot_raw"], index=False)
    chatbot_var.to_csv(paths["chatbot_variance"], index=False)
    agentic_results.to_csv(paths["agentic_raw"], index=False)
    agentic_var.to_csv(paths["agentic_variance"], index=False)
    return {k: str(v) for k, v in paths.items()}


def artifacts(saved_paths: dict[str, str], report_path: str) -> list[Artifact]:
    """Every file this scenario's run reads or produces, for the documentation trail."""
    return [
        Artifact(
            "Custom golden set (input)", "scenarios/fixtures/intended_performance.jsonl",
            "Reused unchanged from scenario 1 — versioned, not regenerated per run.",
        ),
        Artifact(
            "Public benchmark sample (input)", "scenarios/fixtures/public_benchmark_sample.jsonl",
            "Reused unchanged from scenario 1 — versioned, not regenerated per run.",
        ),
        Artifact(
            "Mind2Web task cache (input)", "../Agent/outputs/data/mind2web_train.jsonl",
            "multi_agent_otel_eval's cached copy — outside this repo, in the sibling clone.",
        ),
        Artifact(
            "Chatbot raw results (all 5 repeats)", saved_paths["chatbot_raw"],
            "Every individual run's per-task score, not just the aggregated variance — gitignored, regenerated every run.",
        ),
        Artifact(
            "Chatbot variance by task", saved_paths["chatbot_variance"],
            "The aggregated table shown above, as its own file.",
        ),
        Artifact(
            "Agentic raw results (all 5 repeats × 2 modes)", saved_paths["agentic_raw"],
            "Every individual agent run's score, tool calls, and outcome — gitignored, regenerated every run.",
        ),
        Artifact(
            "Agentic variance by task and mode", saved_paths["agentic_variance"],
            "The aggregated table shown above, as its own file.",
        ),
        Artifact(
            "HTML testing report", report_path,
            "The rendered report embedded above — scope, data, results, charts, and observations.",
        ),
    ]
