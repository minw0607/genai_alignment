"""Scenario-specific logic for Consistency & Reliability (Tier 1).

Kept out of the notebook so the notebook stays a thin, readable narrative —
see notebooks/02_consistency_reliability.ipynb and docs/consistency_reliability.md.

Two tracks, testing two different kinds of "same input, run again":

- Chatbot track: **three** sub-datasets, each run N times. Two are
  unchanged from this scenario's first design — scenario 1 (Intended
  Performance)'s original custom golden set and public-benchmark sample,
  both via genai_capability_bench's generic no-system-prompt adapter. The
  third is new: the same 10 golden-set questions again, but delivered via
  scenario 1's *current* mechanism — its actual RAG-assistant call
  (`scenarios.intended_performance.run_golden_set`, system-prompt mandate +
  per-question knowledge-base document). This scenario deliberately keeps
  both the generic-adapter path and the use-case-grounded path rather than
  retiring the former the way scenario 1 did for its own report — the
  generic path still gives broad, cheap consistency coverage across 40
  varied tasks, and having the *same 10 questions* under two different
  delivery mechanisms turns into a real comparison for free: does adding a
  system-prompt-defined mandate change how consistent the model is, for
  identical questions? "Consistent" = pass/fail doesn't flip across runs
  (raw flag) or fall significantly below an acceptable floor once corrected
  for testing many tasks at once (the rigorous version), and the raw
  answers cluster into one meaning via bidirectional-entailment /
  semantic-entropy, not a TF-IDF similarity threshold.
- Agentic track: a fixed Mind2Web task subset, run N times through
  multi_agent_otel_eval, in both single-agent and multi-agent mode.
  "Consistent" here is NOT text similarity — an agent's actions are compared
  via tool_f1 variance (already equivalence-aware upstream: calling a
  differently-named-but-equivalent tool isn't penalized) and task-outcome
  flip rate, not by diffing raw trajectories.

Generic repeat-run mechanics (combine_repeats, variance_by_task, the
semantic-consistency/Wilson-CI/BH-correction functions) live in
reporting/repeat_run.py, not here, since drift detection will need the same
"run N times, compare" mechanic later. See that module's docstring and
docs/consistency_reliability.md for the statistical methodology and
references (Kuhn et al. 2024; Wilson 1927; Benjamini & Hochberg 1995).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from genai_capability_bench.clients.factory import create_client
from genai_capability_bench.core.schemas import ModelSpec

from adapters.agent_otel import AgentHarness
from adapters.capability_bench import run_capability_scenario
from reporting.artifacts import Artifact
from reporting.display import GENERIC_API_VERSION, GENERIC_JUDGE_MODEL_NAME, GENERIC_MODEL_NAME
from reporting.html_report import ChartImage, DataSection, Metric, ScenarioReport, fig_to_base64
from reporting.repeat_run import (
    add_pairwise_consistency,
    add_reliability_category,
    add_reliability_significance,
    add_semantic_consistency,
    add_wilson_ci,
    combine_repeats,
    repeat_runs,
    variance_by_task,
)
from scenarios import intended_performance as ip_scenario

PALETTE = {"pass": "#2a9d8f", "fail": "#e76f51", "single": "#264653", "multi": "#e9c46a"}

CUSTOM_CONFIG = "configs/intended_performance.yaml"
PUBLIC_CONFIG = "configs/intended_performance_public_benchmark.yaml"
PASS_THRESHOLD = 0.7

CHATBOT_N_REPEATS = 5
AGENTIC_N_REPEATS = 5
AGENTIC_N_TASKS = 5

# Policy choice, not something inferred from data: a task counts as
# acceptably reliable if its true pass rate is at least 80%. Below that,
# add_reliability_significance flags it — corrected for testing 40 (or 10)
# tasks at once via Benjamini-Hochberg, not a raw per-task p-value.
MIN_ACCEPTABLE_PASS_RATE = 0.8

# genai_capability_bench's run_capability_scenario writes each repeat to the
# same fixed run_id directory (configs/intended_performance*.yaml), so calling
# it N times in a loop overwrites itself — only the last repeat would survive
# on disk. This scenario's own output dir is where all N repeats' combined
# raw results actually persist; see save_artifacts().
OUTPUT_DIR = "outputs/runs/consistency_reliability"


def build_judge_client(target_model: str):
    """Client for semantic_consistency's bidirectional-entailment checks —
    same reasoning-family-safe config as scenario 1's judge_review client.
    `target_model` is only a fallback — reads JUDGE_MODEL from the environment
    first, so the judge is a genuinely different model from the one under
    test (see .env.example)."""
    judge_spec = ModelSpec(
        name="judge",
        provider="azure_openai",
        model=os.environ.get("JUDGE_MODEL", target_model),
        temperature=None,
        max_tokens=200,
        token_parameter="max_completion_tokens",
    )
    return create_client(judge_spec)


# ---------------------------------------------------------------- Chatbot track

def run_chatbot_repeats(target_model: str, n: int = CHATBOT_N_REPEATS) -> dict:
    """Run all three chatbot sub-datasets n times each. Returns combined
    long-format results (one row per task per run) tagged with dataset_label.

    `custom_golden_set` and `rag_assistant` share the same 10 underlying
    questions (same task_ids, ip-01..ip-10) delivered through two different
    mechanisms — see module docstring. That's deliberate, but it means
    `chatbot_variance` must group by (task_id, dataset_label), not task_id
    alone, or the two tracks' rows would silently merge."""
    frames = []
    for label, config_path in [
        ("custom_golden_set", CUSTOM_CONFIG),
        ("public_benchmark_sample", PUBLIC_CONFIG),
    ]:
        runs = repeat_runs(lambda cp=config_path: run_capability_scenario(cp)["results"], n, label=label)
        combined = combine_repeats(runs)
        combined["dataset_label"] = label
        frames.append(combined)

    golden_set = ip_scenario.load_golden_set()
    rag_client = ip_scenario.build_target_client(target_model)

    def _run_rag_once() -> pd.DataFrame:
        return ip_scenario.score_golden_set(ip_scenario.run_golden_set(golden_set, rag_client))

    rag_runs = repeat_runs(_run_rag_once, n, label="rag_assistant")
    rag_combined = combine_repeats(rag_runs)
    rag_combined["dataset_label"] = "rag_assistant"
    frames.append(rag_combined)

    return {"results": pd.concat(frames, ignore_index=True), "n_repeats": n}


def chatbot_variance(
    chatbot_results: pd.DataFrame,
    client=None,
    use_semantic_entropy: bool = True,
) -> pd.DataFrame:
    """`client` (see build_judge_client) drives the bidirectional-entailment
    checks inside add_semantic_consistency — required when use_semantic_entropy
    is True (the default). Set use_semantic_entropy=False for a free,
    zero-API-call TF-IDF check instead (add_pairwise_consistency) — faster
    for dev iteration, at the cost of the accuracy this scenario's own
    results demonstrated the entailment method catches: see
    docs/consistency_reliability.md's Sample Results for the specific
    cases (correct-but-differently-worded answers, and vice versa) TF-IDF
    similarity got wrong.

    Grouped by (task_id, dataset_label), not task_id alone — custom_golden_set
    and rag_assistant reuse the same task_ids (ip-01..ip-10) on purpose (see
    run_chatbot_repeats), so dataset_label has to be part of the grouping key
    or their rows would incorrectly merge into one variance row each."""
    id_col = ["task_id", "dataset_label"]
    var = variance_by_task(
        chatbot_results,
        id_col=id_col,
        score_col="score",
        passed_col="passed",
        passthrough_cols=["category"],
    )
    if use_semantic_entropy:
        if client is None:
            raise ValueError("client is required when use_semantic_entropy=True")
        var = add_semantic_consistency(var, chatbot_results, id_col=id_col, text_col="actual_output", client=client)
    else:
        var = add_pairwise_consistency(var, chatbot_results, id_col=id_col, text_col="actual_output")
    var = add_wilson_ci(var)
    var = add_reliability_significance(var, min_acceptable_rate=MIN_ACCEPTABLE_PASS_RATE)
    var = add_reliability_category(var)
    return var


_DATASET_TAG = {"custom_golden_set": "custom", "rag_assistant": "rag", "public_benchmark_sample": "public"}


def plot_chatbot_variance(variance: pd.DataFrame) -> ChartImage:
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))

    ordered = variance.sort_values(["dataset_label", "task_id"]).copy()
    # custom_golden_set and rag_assistant share task_ids (ip-01..ip-10) on
    # purpose (see run_chatbot_repeats) — tag every label with its dataset so
    # the two tracks' bars for "the same question" aren't mistaken for one.
    short_id = ordered["task_id"].str.replace("answer_accuracy_knowledge_v1_", "", regex=False)
    labels = ordered["dataset_label"].map(_DATASET_TAG).fillna(ordered["dataset_label"]) + ":" + short_id

    colors = ordered["flips"].map({True: PALETTE["fail"], False: PALETTE["pass"]})
    axes[0].bar(range(len(ordered)), ordered["score_std"], color=colors)
    axes[0].set_xticks(range(len(ordered)))
    axes[0].set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    axes[0].set_ylabel("score std dev across runs")
    axes[0].set_title("Score variance by task (red = pass/fail flipped)")

    axes[1].bar(range(len(ordered)), ordered["semantic_consistency"], color=PALETTE["single"])
    axes[1].set_xticks(range(len(ordered)))
    axes[1].set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("1 − normalized semantic entropy")
    axes[1].set_title("Semantic consistency of raw answers across runs")

    plt.tight_layout()
    chart = ChartImage(
        title="Chatbot repeat-run variance",
        caption=(
            "Left: how much each task's score moved across repeated runs (red bars flipped "
            "pass/fail at least once). Right: whether the N raw answers cluster into one meaning "
            "or several, via bidirectional-entailment clustering (Kuhn et al. 2024) rather than "
            "lexical similarity — a task can pass every run and still show clusters here if it "
            "means something different each time (or score 1.0 while phrasing it differently, "
            "since entailment — unlike TF-IDF — recognizes paraphrase as equivalent). Labels are "
            "tagged `custom:`/`rag:`/`public:` — `custom` and `rag` are the *same 10 questions*, "
            "delivered two different ways (see Methodology); compare them directly."
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
        # verbose=True turns on AgentHarness.run's own per-task progress line
        # (adapters/agent_otel.py) — each repeat here is a full task batch
        # that can take minutes, so both levels of progress matter.
        runs = repeat_runs(lambda t=tasks, m=mode: harness.run(t, mode=m, verbose=True), n_repeats, label=f"{mode}-agent")
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
    var = variance_by_task(
        agentic_results,
        id_col=["task_id", "mode"],
        score_col="task_score",
        passed_col="task_passed",
        extra_numeric_cols=["tool_f1", "n_tool_calls"],
        passthrough_cols=["website"],
    )
    var = add_wilson_ci(var)
    var = add_reliability_significance(var, min_acceptable_rate=MIN_ACCEPTABLE_PASS_RATE)
    var = add_reliability_category(var)
    return var


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
    counts = chatbot_var["reliability_category"].value_counts()
    n_unstable = int(counts.get("unstable", 0))
    n_consistently_failing = int(counts.get("consistently_failing", 0))
    obs.append(
        f"{n_flips} of {len(chatbot_var)} tasks showed *any* pass/fail disagreement across "
        f"{CHATBOT_N_REPEATS} repeated runs, but a raw flip and a real consistency finding aren't the "
        f"same claim. After Benjamini-Hochberg correction against an {MIN_ACCEPTABLE_PASS_RATE:.0%} "
        f"reliability floor: **{n_unstable} genuinely unstable** (significant *and* actually flipped — "
        f"this is what this scenario tests for) and **{n_consistently_failing} consistently failing** "
        "(significant, but never flipped — the same wrong or under-scored answer every single run, "
        "which is a capability-gap or scoring-artifact finding riding along on the same statistical "
        "test, not instability). Check `semantic_consistency` on the consistently-failing rows: near "
        "1.0 means the model confidently gives the same answer every time — wrong every time is a "
        "capability gap, and worth a judge check (as scenario 1 does) to rule out a scoring-threshold "
        "artifact before calling it that."
    )
    low_meaning_consistency = chatbot_var[chatbot_var["semantic_consistency"] < 0.99]
    high_cluster_low_flip = low_meaning_consistency[~low_meaning_consistency["flips"]]
    if len(high_cluster_low_flip):
        obs.append(
            f"{len(high_cluster_low_flip)} task(s) passed every run yet still split into more than "
            "one meaning-cluster under bidirectional-entailment scoring — the model reaches a "
            "*passing* score every time but doesn't always mean the same thing when it does. "
            "Pass-rate alone would have missed this entirely; it's exactly the case semantic "
            "consistency (Kuhn et al. 2024) is meant to catch and TF-IDF similarity was not "
            "reliably catching before."
        )

    custom = chatbot_var[chatbot_var["dataset_label"] == "custom_golden_set"]
    rag = chatbot_var[chatbot_var["dataset_label"] == "rag_assistant"]
    if len(custom) and len(rag):
        obs.append(
            f"**Same 10 questions, two delivery mechanisms — a real comparison, not just extra "
            f"coverage:** `custom_golden_set` (generic adapter, no system prompt) averages "
            f"{custom['semantic_consistency'].mean():.2f} semantic consistency and "
            f"{custom['score_std'].mean():.2f} avg score std dev; `rag_assistant` (system-prompt "
            f"mandate + knowledge-base document, scenario 1's current mechanism) averages "
            f"{rag['semantic_consistency'].mean():.2f} and {rag['score_std'].mean():.2f} on the "
            "identical questions. A material gap either direction would mean the mandate framing "
            "itself changes how consistent the model is, not just what it scores — read the two "
            "`reliability_category` columns side by side in the results table below before "
            "concluding either way from the averages alone."
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
        counts = group["reliability_category"].value_counts()
        n_unstable = int(counts.get("unstable", 0))
        n_consistently_failing = int(counts.get("consistently_failing", 0))
        detail = (
            f"average tool_f1 std dev {group['tool_f1_std'].mean():.2f}"
            if not tool_f1_flat
            else f"average n_tool_calls std dev {group['n_tool_calls_std'].mean():.1f}"
        )
        obs.append(
            f"{mode}-agent: {n_flips} of {len(group)} tasks flipped pass/fail across "
            f"{AGENTIC_N_REPEATS} repeated runs — {n_unstable} genuinely unstable, "
            f"{n_consistently_failing} consistently failing after BH correction at a "
            f"{MIN_ACCEPTABLE_PASS_RATE:.0%} reliability floor (with only 5 tasks per mode, this "
            "correction has little power, so treat these as weak signals, not settled findings); "
            f"{detail}."
        )
    return obs


def _high_risk_cases(chatbot_var: pd.DataFrame, agentic_var: pd.DataFrame) -> list[str]:
    """The specific tasks this scenario actually exists to catch: genuine
    run-to-run instability, not a consistently-failing capability gap or
    scoring artifact riding along on the same significance test. Derived
    dynamically from this run's own reliability_category, never hardcoded."""
    cases = []
    for _, row in chatbot_var[chatbot_var["reliability_category"] == "unstable"].iterrows():
        cases.append(
            f"Chatbot `{row['dataset_label']}` / `{row['task_id']}`: pass rate {row['pass_rate']:.0%} across "
            f"{int(row['n_runs'])} runs (flipped pass/fail), semantic consistency "
            f"{row['semantic_consistency']:.2f} — genuinely unstable, not a scoring-threshold artifact."
        )
    for _, row in agentic_var[agentic_var["reliability_category"] == "unstable"].iterrows():
        cases.append(
            f"Agentic `{row['mode']}`-agent task `{row['task_id']}`: pass rate {row['pass_rate']:.0%} across "
            f"{int(row['n_runs'])} runs (flipped outcome), tool_f1 std dev {row['tool_f1_std']:.2f} — "
            "genuinely unstable."
        )
    return cases


def build_report(
    chatbot_var: pd.DataFrame,
    agentic_var: pd.DataFrame,
    target_model: str,
    api_version: str,
    charts: list[ChartImage],
    artifacts_table: pd.DataFrame | None = None,
) -> ScenarioReport:
    overall_flip_rate_chat = chatbot_var["flips"].mean()
    overall_flip_rate_agent = agentic_var["flips"].mean()

    n_unstable_chat = int((chatbot_var["reliability_category"] == "unstable").sum())
    n_failing_chat = int((chatbot_var["reliability_category"] == "consistently_failing").sum())
    n_unstable_agent = int((agentic_var["reliability_category"] == "unstable").sum())
    n_failing_agent = int((agentic_var["reliability_category"] == "consistently_failing").sum())

    executive_summary = (
        f"This run tested whether the same input produces the same output across {CHATBOT_N_REPEATS} "
        f"repeats (chatbot track, {len(chatbot_var)} task/dataset combinations across three sub-datasets — "
        "a generic no-system-prompt path and the same questions again via scenario 1's actual RAG-assistant "
        f"mechanism) and {AGENTIC_N_REPEATS} repeats (agentic track, {len(agentic_var)} task/mode "
        "combinations, single- and multi-agent Mind2Web navigation). After correcting for testing many "
        f"tasks at once (Benjamini-Hochberg): chatbot — {n_unstable_chat} genuinely unstable, "
        f"{n_failing_chat} consistently failing (a capability gap or scoring artifact, not instability); "
        f"agentic — {n_unstable_agent} genuinely unstable, {n_failing_agent} consistently failing. "
        + (
            "See High-Risk Cases for the specific genuinely-unstable tasks."
            if (n_unstable_chat + n_unstable_agent) else
            "No task in either track showed genuine run-to-run instability this run — every significant "
            "finding was a repeatable capability gap or scoring artifact, not inconsistency."
        )
    )

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
        "Increase agentic repeats beyond 5 — independent re-runs under the same judge have "
        "classified different specific tasks as unstable vs. consistently_failing, direct "
        "evidence n=5 is too small for stable per-task categorization on this track.",
        "Build a controlled ablation (same task, orchestration disabled vs. enabled) to actually "
        "isolate how much variance the harness adds on top of the model's own — the honest gap "
        "flagged in the observations above.",
        "The bidirectional-entailment check uses an LLM-judge prompt (JUDGE_MODEL, a different "
        "deployment than the target model) rather than a dedicated NLI model. A dedicated NLI "
        "model (e.g. DeBERTa-MNLI, as Kuhn et al. 2024 use) would remove LLM-judge variance from "
        "the entailment check itself, at the cost of a new ML dependency.",
        f"{MIN_ACCEPTABLE_PASS_RATE:.0%} is a stated policy choice for the reliability floor, not "
        "something derived from data — revisit it once there's a real SLA or business requirement "
        "to calibrate against.",
    ]

    return ScenarioReport(
        scenario_name="Consistency & Reliability",
        tier="Tier 1",
        risk="Same input yields different output; hallucination / variance.",
        goal="Repeatable, consistent outputs for equivalent inputs.",
        target_summary={
            "Provider": "Azure OpenAI",
            "Model": GENERIC_MODEL_NAME,  # never the real deployment name — see reporting/display.py
            "API version": GENERIC_API_VERSION,  # never the real value — see reporting/display.py
            "Judge model": GENERIC_JUDGE_MODEL_NAME if os.environ.get("JUDGE_MODEL") else "<falls back to target model>",
            "Chatbot track": (
                f"{CHATBOT_N_REPEATS} repeats each of custom_golden_set, public_benchmark_sample "
                "(scenario 1's original generic-adapter path), and rag_assistant (scenario 1's "
                "current RAG-assistant mechanism, same 10 questions as custom_golden_set)"
            ),
            "Agentic track": (
                f"{AGENTIC_N_REPEATS} repeats of {AGENTIC_N_TASKS} Mind2Web tasks, "
                "single-agent and multi-agent mode"
            ),
        },
        approach=(
            "Two tracks, testing two different kinds of 'same input, run again.' Chatbot: three "
            "sub-datasets, run N times via `reporting/repeat_run.py`'s generic repeat harness — "
            "`custom_golden_set` and `public_benchmark_sample` reuse scenario 1's *original* "
            "generic-adapter mechanism unchanged (no system prompt); `rag_assistant` repeats the "
            "same 10 questions as `custom_golden_set` again, but via scenario 1's *current* native "
            "RAG-assistant call (imported directly from `scenarios.intended_performance`) — kept "
            "deliberately alongside the generic path rather than replacing it, so this scenario has "
            "both broad generic-adapter coverage and a direct same-questions comparison of whether "
            "a system-prompt-defined mandate changes consistency. Agentic: a new adapter "
            "(`adapters/agent_otel.py`) onto `multi_agent_otel_eval`'s `evaluate_batch()`, run N "
            "times on a fixed Mind2Web task subset in both single- and multi-agent mode. Both "
            "tracks share the same base variance function (`variance_by_task`) plus two "
            "statistical layers on top — Wilson confidence intervals on the pass rate, and "
            "Benjamini-Hochberg-corrected significance testing against a stated reliability floor "
            "— configured with different column names rather than duplicated. The chatbot track "
            "additionally adds semantic consistency (bidirectional-entailment clustering, Kuhn et "
            "al. 2024) on the raw text answers; the agentic track adds tool_f1 variance instead, "
            "since comparing raw agent trajectories as text wouldn't be the right notion of "
            "'consistent action.'"
        ),
        data_sections=[
            DataSection(
                name="Chatbot: custom_golden_set + public_benchmark_sample",
                layer="Reused, not new",
                source="Scenario 1's original custom HR/IT policy set + public MMLU/TriviaQA/ARC sample",
                size=f"40 tasks × {CHATBOT_N_REPEATS} repeats = {40 * CHATBOT_N_REPEATS} calls",
                description="Identical data and mechanism to scenario 1's first design — this scenario only adds the repeat dimension.",
            ),
            DataSection(
                name="Chatbot: rag_assistant",
                layer="Reused data, new delivery mechanism",
                source="Same 10 questions as custom_golden_set, via scenario 1's current RAG-assistant call",
                size=f"10 tasks × {CHATBOT_N_REPEATS} repeats = {10 * CHATBOT_N_REPEATS} calls",
                description="The same-questions-two-mechanisms comparison — see Approach.",
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
            Metric(
                value=f"{int((chatbot_var[chatbot_var['dataset_label']=='rag_assistant']['reliability_category'] == 'consistently_failing').sum())}/10",
                label="rag_assistant: consistently failing",
                sublabel="same 10 questions as custom_golden_set, native RAG mechanism",
            ),
            Metric(
                value=f"{int((chatbot_var[chatbot_var['dataset_label']=='custom_golden_set']['reliability_category'] == 'consistently_failing').sum())}/10",
                label="custom_golden_set: consistently failing",
                sublabel="identical questions, generic adapter (no system prompt)",
            ),
            Metric(
                value=f"{int((chatbot_var['reliability_category'] == 'unstable').sum())}/{len(chatbot_var)}",
                label="Chatbot tasks genuinely unstable",
                sublabel=f"{overall_flip_rate_chat:.0%} raw flip rate, {len(chatbot_var)} tasks across all 3 sub-datasets",
            ),
            Metric(
                value=f"{int((chatbot_var['reliability_category'] == 'consistently_failing').sum())}/{len(chatbot_var)}",
                label="Chatbot tasks consistently failing (all sub-datasets)",
                sublabel="capability gap or scoring artifact, not instability",
            ),
            Metric(
                value=f"{int((agentic_var['reliability_category'] == 'unstable').sum())}/{len(agentic_var)}",
                label="Agentic tasks genuinely unstable",
                sublabel=f"{overall_flip_rate_agent:.0%} raw flip rate, {len(agentic_var)} task/modes tested",
            ),
            Metric(
                value=f"{int((agentic_var['reliability_category'] == 'consistently_failing').sum())}/{len(agentic_var)}",
                label="Agentic tasks consistently failing",
                sublabel="capability gap or scoring artifact, not instability",
            ),
            Metric(value=f"{chatbot_var['semantic_consistency'].mean():.2f}", label="Avg chatbot semantic consistency"),
            Metric(value=f"{agentic_var['tool_f1_std'].mean():.2f}", label="Avg agentic tool_f1 std dev"),
        ],
        results_tables=[
            ("Chatbot variance by task", chatbot_var),
            ("Agentic variance by task and mode", agentic_var),
        ],
        charts=charts,
        executive_summary=executive_summary,
        observations=observations,
        high_risk_cases=_high_risk_cases(chatbot_var, agentic_var),
        next_steps=next_steps,
        artifacts_table=artifacts_table,
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


def artifacts(saved_paths: dict[str, str]) -> list[Artifact]:
    """Every file this scenario's run reads or produces, rendered into the
    report's own Appendix (built before the report itself is saved, so the
    report file isn't listed here — see scenarios/adversarial_inputs.py for
    the same pattern)."""
    return [
        Artifact(
            "Custom golden set (input)", "scenarios/fixtures/intended_performance.jsonl",
            "Reused from scenario 1 — same file drives both custom_golden_set and rag_assistant.",
        ),
        Artifact(
            "Public benchmark sample (input)", "scenarios/fixtures/public_benchmark_sample.jsonl",
            "Reused unchanged from scenario 1's original design — versioned, not regenerated per run.",
        ),
        Artifact(
            "Mind2Web task cache (input)", "../Agent/outputs/data/mind2web_train.jsonl",
            "multi_agent_otel_eval's cached copy — outside this repo, in the sibling clone.",
        ),
        Artifact(
            "Chatbot raw results (all repeats, all 3 sub-datasets)", saved_paths["chatbot_raw"],
            "Every individual run's per-task score, not just the aggregated variance — gitignored, regenerated every run.",
        ),
        Artifact(
            "Chatbot variance by task", saved_paths["chatbot_variance"],
            "The aggregated table shown above, as its own file.",
        ),
        Artifact(
            "Agentic raw results (all repeats × 2 modes)", saved_paths["agentic_raw"],
            "Every individual agent run's score, tool calls, and outcome — gitignored, regenerated every run.",
        ),
        Artifact(
            "Agentic variance by task and mode", saved_paths["agentic_variance"],
            "The aggregated table shown above, as its own file.",
        ),
    ]
