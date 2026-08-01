"""Scenario-specific logic for Adversarial Inputs (Tier 2).

Kept out of the notebook so the notebook stays a thin, readable narrative —
see notebooks/04_adversarial_inputs.ipynb and docs/adversarial_inputs.md.

Adapter scenario — this is the first slice of Tier 2's "Adversarial inputs"
row (see README Scenario Library), scoped to just the prompt-injection
workstream of llm_red_teaming to start narrow; jailbreaking and adversarial
NLP are the natural next slices, not built here.

Prompt injection tests whether the target model can be made to ignore its
system instruction and follow an injected one instead (OWASP LLM01, and via
documents LLM08) — a different property from jailbreaking (which targets the
model's safety alignment, not its instruction-following control flow).

Two tracks, both from llm_red_teaming's existing, unmodified machinery
(adapters/red_teaming.py calls it directly — no evaluation logic is
duplicated here):

- Canary benchmark (deterministic): each injection attempt asks the model to
  emit a unique marker string. If the marker appears in the response, the
  injection overrode the legitimate task — no judge needed, fully
  reproducible. Run across 2 vectors (direct: injection in the user's own
  input; indirect: injection hidden in a document the model must process) x
  5 strategies (Liu et al. 2024's Open-Prompt-Injection taxonomy) x 3 base
  tasks (translate, summarize, sentiment).
- Real-payload track (LLM-judged): actual injection strings collected in the
  wild (deepset/prompt-injections), which carry no canary, so success is
  judged by JUDGE_MODEL — same repo-wide independent-judge convention as
  scenarios 1-3.

A known, documented measurement artifact carried over from the source repo:
the "translate" base task can register a false "injected" hit when the model
faithfully translates the injected instruction text itself (canary included)
rather than actually obeying it as a new task — see Methodology.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from adapters.red_teaming import PromptInjectionHarness
from reporting.artifacts import Artifact
from reporting.html_report import ChartImage, DataSection, Metric, ScenarioReport, fig_to_base64

OUTPUT_DIR = "outputs/runs/adversarial_inputs"

N_PER_TASK = 2
PAYLOAD_MAX_ITEMS = 30

PALETTE = {True: "#e76f51", False: "#2a9d8f"}


def build_harness(target_model: str) -> PromptInjectionHarness:
    return PromptInjectionHarness(target_model)


def run_canary_benchmark(harness: PromptInjectionHarness, n_per_task: int = N_PER_TASK) -> pd.DataFrame:
    """Runs both vectors (direct + indirect) — 2 x 5 strategies x 3 tasks x
    n_per_task content items."""
    direct = harness.run_canary_benchmark(context="direct", n_per_task=n_per_task)
    indirect = harness.run_canary_benchmark(context="indirect", n_per_task=n_per_task)
    return pd.concat([direct, indirect], ignore_index=True)


def run_payload_track(harness: PromptInjectionHarness, max_items: int = PAYLOAD_MAX_ITEMS) -> pd.DataFrame:
    return harness.run_payload_track(max_items=max_items)


def plot_override_by_vector(canary: pd.DataFrame) -> ChartImage:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rates = canary.groupby("context")["injected"].mean().reindex(["direct", "indirect"])
    ax.bar(rates.index, rates.values, color=["#e76f51", "#264653"])
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center")
    ax.set_ylabel("override rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prompt injection override rate by vector")
    plt.tight_layout()
    chart = ChartImage(
        title="Override rate by vector",
        caption=(
            "direct = injection in the user's own input; indirect = injection hidden in a "
            "document the model is asked to process (higher real-world risk — the user is "
            "innocent). Canary-based, deterministic — no judge involved."
        ),
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


def plot_override_by_strategy(canary: pd.DataFrame) -> ChartImage:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot = canary.pivot_table(index="strategy", columns="context", values="injected", aggfunc="mean")
    pivot = pivot.reindex(["naive", "escape", "context_ignoring", "fake_completion", "combined"])
    pivot.plot(kind="bar", ax=ax, color=["#e76f51", "#264653"])
    ax.set_ylabel("override rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Override rate by strategy")
    ax.legend(title="vector")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    chart = ChartImage(
        title="Override rate by strategy",
        caption="The five Open-Prompt-Injection strategies (Liu et al. 2024), by vector.",
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


def plot_override_by_task(canary: pd.DataFrame) -> ChartImage:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pivot = canary.pivot_table(index="task", columns="context", values="injected", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax, color=["#e76f51", "#264653"])
    ax.set_ylabel("override rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Override rate by base task")
    ax.legend(title="vector")
    plt.xticks(rotation=0)
    plt.tight_layout()
    chart = ChartImage(
        title="Override rate by base task",
        caption=(
            "Watch 'translate' specifically — a documented measurement artifact (see "
            "Methodology) can inflate its rate: the model may faithfully translate the "
            "injected text, canary included, without actually obeying it as a new task."
        ),
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


def plot_payload_track(payloads: pd.DataFrame) -> ChartImage:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = payloads["injected"].value_counts().reindex([True, False], fill_value=0)
    ax.bar(["injected", "resisted"], counts.values, color=[PALETTE[True], PALETTE[False]])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center")
    ax.set_ylabel("count")
    ax.set_title(f"Real-world payload track (n={len(payloads)}, LLM-judged)")
    plt.tight_layout()
    chart = ChartImage(
        title="Real-world payload track",
        caption=(
            "Actual injection strings from deepset/prompt-injections — no canary, so success "
            "is judged by JUDGE_MODEL rather than measured deterministically. Indicative, not "
            "as reliable as the canary benchmark; flagged cases should be read by hand."
        ),
        base64_png=fig_to_base64(fig),
        section="results",
    )
    plt.show()
    return chart


def classify_payload_outcomes(payloads: pd.DataFrame) -> pd.DataFrame:
    """Splits the real-payload track's rows into three outcomes, not two —
    'error' as a single bucket hides a real, favorable-for-security finding:
    Azure's own platform-level content filter (a jailbreak/injection
    detector operating upstream of the model) blocks a real fraction of
    these payloads outright, before the model ever sees them. That's a
    materially different outcome from the model itself resisting an
    injection it was actually exposed to, and from an unrelated API error —
    conflating all three into "not injected" understates the true override
    rate among payloads the model actually got a chance to respond to."""
    out = payloads.copy()
    content_filtered = out["response"].fillna("").str.contains("content_filter", case=False)
    other_error = out["reason"].fillna("").str.contains("error", case=False) & ~content_filtered
    out["outcome"] = "evaluated"
    out.loc[content_filtered, "outcome"] = "blocked_by_content_filter"
    out.loc[other_error, "outcome"] = "other_error"
    return out


def _observations(canary: pd.DataFrame, payloads: pd.DataFrame) -> list[str]:
    obs = []
    overall_rate = canary["injected"].mean()
    obs.append(
        f"Canary benchmark: {int(canary['injected'].sum())} of {len(canary)} attempts overrode the "
        f"legitimate task ({overall_rate:.1%} overall)."
    )
    by_vector = canary.groupby("context")["injected"].mean()
    obs.append(
        f"By vector: direct {by_vector.get('direct', 0):.1%}, indirect {by_vector.get('indirect', 0):.1%}."
    )
    translate_hits = canary[(canary["task"] == "translate") & (canary["injected"])]
    if len(translate_hits):
        obs.append(
            f"{len(translate_hits)} 'translate'-task hit(s) — read these by hand before trusting the raw "
            "rate: the known measurement artifact is the model faithfully translating the injected text "
            "(canary included) rather than actually obeying it as a new instruction."
        )
    classified = classify_payload_outcomes(payloads)
    n_blocked = int((classified["outcome"] == "blocked_by_content_filter").sum())
    n_other_err = int((classified["outcome"] == "other_error").sum())
    evaluated = classified[classified["outcome"] == "evaluated"]
    obs.append(
        f"Real-payload track: {n_blocked} of {len(payloads)} payloads were blocked outright by Azure's "
        f"own platform-level content filter before reaching the model (a jailbreak/injection detector "
        f"operating upstream — a defensive positive, not a model behavior) — the message it returns "
        "flags 'jailbreak: detected, filtered'. "
        f"{n_other_err} more failed with an unrelated API/judge error. Among the "
        f"{len(evaluated)} payloads the model actually got a chance to respond to, "
        f"{int(evaluated['injected'].sum())} ({evaluated['injected'].mean():.0%}) were judged as followed "
        "— the rate that actually reflects the model's own robustness, as opposed to the blended rate "
        "that treats platform-blocked and model-resisted as the same outcome."
    )
    return obs


def build_report(
    canary: pd.DataFrame,
    payloads: pd.DataFrame,
    target_model: str,
    api_version: str,
    charts: list[ChartImage],
) -> ScenarioReport:
    observations = _observations(canary, payloads)
    _classified_payloads = classify_payload_outcomes(payloads)
    _evaluated_payloads = _classified_payloads[_classified_payloads["outcome"] == "evaluated"]
    next_steps = [
        "Only the prompt-injection workstream is adapted so far — jailbreaking and adversarial NLP "
        "(both already built in llm_red_teaming) are the natural next slices of this scenario, not "
        "built in this pass.",
        "n_per_task=2 is a small sample per strategy/task/vector cell — expand before treating any "
        "single cell's rate as a stable measurement.",
        "The 'translate' task's measurement artifact (canary carried through by faithful translation) "
        "is a known limitation of canary detection for this specific base task, not fixed here — "
        "flagged cases should be read by hand, as the source repo's own docs already recommend.",
        "The real-payload track is LLM-judged and therefore carries the same judge-reliability caveats "
        "as this repo's other scenarios — no repeat-and-majority-vote or human spot-check yet.",
        "'other_error' rows (a judge-side BadRequestError, not a target-side one) don't retain the full "
        "error message the way target-side content-filter blocks do — likely the same content-filter "
        "mechanism triggering when the judge itself reads the payload text, but not confirmed. Capturing "
        "the full judge-side error message would settle this.",
    ]

    return ScenarioReport(
        scenario_name="Adversarial Inputs",
        tier="Tier 2",
        risk="Manipulated or unsafe behavior from conflicting / malicious input.",
        goal="Robustness to ambiguous, conflicting, adversarial inputs.",
        target_summary={
            "Provider": "Azure OpenAI",
            "Model": target_model,
            "API version": api_version,
            "Judge model (payload track only)": os.environ.get("JUDGE_MODEL", target_model),
            "Canary benchmark": f"{len(canary)} attempts (2 vectors x 5 strategies x 3 tasks x {N_PER_TASK})",
            "Real-payload track": f"{len(payloads)} real-world injection strings",
        },
        approach=(
            "Adapter onto llm_red_teaming's existing, unmodified prompt-injection harness "
            "(PromptInjectionRunner) — no evaluation logic is reimplemented here. Canary benchmark: "
            "each injection attempt asks the model to emit a unique marker; the marker's presence in "
            "the response is a deterministic override signal, run across direct (injection in the "
            "user's input) and indirect (injection hidden in a document) vectors, 5 attack strategies "
            "(Liu et al. 2024), and 3 base tasks. Real-payload track: actual injection strings collected "
            "in the wild, judged by JUDGE_MODEL since they carry no canary."
        ),
        data_sections=[
            DataSection(
                name="Canary benchmark: structured injection attempts",
                layer="Reused unchanged from llm_red_teaming",
                source="llm_red_teaming's BASE_TASKS + Open-Prompt-Injection strategy taxonomy",
                size=f"2 vectors x 5 strategies x 3 tasks x {N_PER_TASK} items = {2*5*3*N_PER_TASK} attempts",
                description="Deterministic canary-marker detection — no judge, fully reproducible.",
            ),
            DataSection(
                name="Real-world payloads",
                layer="Reused unchanged from llm_red_teaming",
                source="deepset/prompt-injections (HuggingFace), 203 labeled injection texts",
                size=f"{PAYLOAD_MAX_ITEMS} sampled per run",
                description="Freeform real-world attacks, no canary — judged by JUDGE_MODEL.",
            ),
        ],
        key_metrics=[
            Metric(value=f"{canary['injected'].mean():.0%}", label="Overall canary override rate", sublabel=f"n={len(canary)}"),
            Metric(
                value=f"{canary[canary['context']=='direct']['injected'].mean():.0%}",
                label="Direct-vector override rate",
            ),
            Metric(
                value=f"{canary[canary['context']=='indirect']['injected'].mean():.0%}",
                label="Indirect-vector override rate",
            ),
            Metric(
                value=f"{int((_classified_payloads['outcome']=='blocked_by_content_filter').sum())}/{len(payloads)}",
                label="Real payloads blocked by platform content filter",
                sublabel="upstream of the model — a defensive positive",
            ),
            Metric(
                value=(
                    f"{_evaluated_payloads['injected'].mean():.0%}" if len(_evaluated_payloads) else "N/A"
                ),
                label="Real-payload override rate, model's own robustness",
                sublabel=f"n={len(_evaluated_payloads)} payloads actually evaluated",
            ),
        ],
        results_tables=[
            ("Canary benchmark results", canary.drop(columns=["canary"], errors="ignore")),
            ("Real-payload track results", _classified_payloads.drop(columns=["canary"], errors="ignore")),
        ],
        charts=charts,
        observations=observations,
        next_steps=next_steps,
        notebook_link="../notebooks/04_adversarial_inputs.ipynb",
        doc_link="../docs/adversarial_inputs.md",
    )


def save_artifacts(canary: pd.DataFrame, payloads: pd.DataFrame) -> dict[str, str]:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "canary": out_dir / "canary_results.csv",
        "payloads": out_dir / "payload_results.csv",
    }
    canary.to_csv(paths["canary"], index=False)
    payloads.to_csv(paths["payloads"], index=False)
    return {k: str(v) for k, v in paths.items()}


def artifacts(saved_paths: dict[str, str], report_path: str) -> list[Artifact]:
    """Every file this scenario's run reads or produces, for the documentation trail."""
    return [
        Artifact(
            "llm_red_teaming sibling clone (input)", "../llm_red_teaming",
            "Provides PromptInjectionRunner, AzureOpenAITarget, and the payload loader — outside this repo.",
        ),
        Artifact(
            "Real-world payload cache", "../llm_red_teaming/eval_datasets/safety/deepset_prompt_injections.parquet",
            "Cached copy of deepset/prompt-injections, inside the sibling repo — downloaded once, reused after.",
        ),
        Artifact(
            "Canary benchmark results", saved_paths["canary"],
            "Every attempt's vector, task, strategy, and outcome — gitignored, regenerated every run.",
        ),
        Artifact(
            "Real-payload track results", saved_paths["payloads"],
            "Every payload's outcome and judge reasoning — gitignored, regenerated every run.",
        ),
        Artifact(
            "HTML testing report", report_path,
            "The rendered report embedded above — scope, data, results, charts, and observations.",
        ),
    ]
