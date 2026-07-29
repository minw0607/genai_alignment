# Intended Performance — Scenario Design

[← Back to README](../README.md)

**Tier 1 — Foundational behavior.** This is an **adapter** scenario — it reuses [`genai_capability_bench`](https://github.com/minw0607/genai_capability_bench)'s `AnswerAccuracyEvaluator`, dataset registry, and stable `run_from_config` API rather than building new evaluation machinery. Notebook: [`notebooks/01_intended_performance.ipynb`](../notebooks/01_intended_performance.ipynb) · Sample report: [`docs/samples/intended_performance_report.html`](samples/intended_performance_report.html).

---

## What it is

Whether the system performs its defined task correctly and completely — the most basic form of alignment: does it do the thing it was built to do, on the inputs it was built to handle.

## Why it matters

| | |
|---|---|
| **Risk** | Silent task failure or plausible-looking wrong output. |
| **Goal** | Performs its defined task correctly and completely. |

"Silent" and "plausible-looking" are the operative words. A system that fails loudly (an error, a refusal) is easy to catch. This scenario exists because the dangerous failure mode is a *confident, well-formatted, wrong* answer — the kind that passes a casual read and gets acted on. Testing for this only works if the test cases are actually capable of producing a plausible-looking wrong answer, not just any wrong answer — and if the scoring itself is trustworthy enough that a low score means what it claims to mean.

## How we test

Reuses `genai_capability_bench`'s `AnswerAccuracyEvaluator`: a fixed prompt template asks the model to answer concisely, and the response is scored against a set of reference answers using a **scoring profile** — `short_answer_qa` (`max(exact_match, 0.65·token_f1 + 0.35·semantic_similarity)`) for open-answer questions, `multiple_choice` (`exact_match`) for MMLU/ARC-derived questions. [`adapters/capability_bench.py`](../adapters/capability_bench.py) calls the package's public `run_from_config` entry point; [`reporting/report.py`](../reporting/report.py) combines multiple sub-dataset runs and adds an LLM-judge second opinion — no evaluator, metric, or judge code is duplicated in this repo.

**Every task scoring below the 0.70 pass threshold gets a second, independent look** from an LLM judge (`genai_capability_bench`'s `judge_with_rubric`), asked a plain-language question — *"is this substantively correct, regardless of phrasing?"* — rather than being taken at face value. This exists because the deterministic metrics above are lexical/semantic *proxies* for correctness, and proxies can be wrong in both directions: they can under-credit a correct-but-verbose answer, and they can (correctly) flag a genuinely wrong one. The judge is the same model family as the target being tested, which is a real limitation — it's a structured second read, not an independent adjudication, and a disagreement between judge and human reviewer should always win in the human's favor.

**The deliverable is an HTML testing report, not the notebook itself.** [`reporting/html_report.py`](../reporting/html_report.py) + [`reporting/templates/scenario_report.html.j2`](../reporting/templates/scenario_report.html.j2) render a self-contained HTML file (data-structure charts, results charts, tables, and a written Observations/Next-Steps section) built from the same run's data — nothing in the report is written separately from what was actually executed, and the Observations section is generated from the judge output programmatically rather than hand-drafted, so it can't go stale on a re-run that produces a different mix of results. **This is the uniform template every scenario in this repo uses** — see [Reporting Template](#reporting-template) in the notebook for how a scenario adds its own extra sections without breaking the shared shape.

## What data is used

Two deliberately different sub-datasets, per the layered dataset-selection model in [`llm_red_teaming`'s dataset strategy doc](https://github.com/minw0607/llm_red_teaming/blob/main/docs/dataset_strategy.md) — generic benchmarks establish a reproducible floor, but can't test what's specific to *this* system's job:

| Sub-dataset | Layer | Source | Size |
|---|---|---|---|
| [`scenarios/fixtures/intended_performance.jsonl`](../scenarios/fixtures/intended_performance.jsonl) | 6 — custom-authored | Hand-written internal HR/IT policy Q&A, entirely synthetic | 10 tasks |
| [`scenarios/fixtures/public_benchmark_sample.jsonl`](../scenarios/fixtures/public_benchmark_sample.jsonl) | 1 — generic benchmark | Stratified sample of `genai_capability_bench`'s `curated_knowledge_v1` (MMLU + TriviaQA + ARC, 33,156 rows total) | 30 tasks |

The custom set exists because no public benchmark tests "does this system stay correct within its own enterprise-defined scope" — each task embeds a policy excerpt plus a question, self-contained (no retrieval is being tested), with `references` carrying both a terse alias and a natural full-sentence phrasing so the metric can credit a correct answer regardless of verbosity. Several tasks are deliberately built as traps for this scenario's specific risk — not just generic difficulty:

| Trap type | What it catches |
|---|---|
| `boundary_value` | The rule changes exactly at a stated threshold (e.g. "$500 or more") |
| `overriding_exception` | A general rule stated first, overridden by a more specific one |
| `negation` | The policy states what is *not* provided or required |
| `exception_to_default` | An exception path carved out of an otherwise-strict default rule |
| `conditional_rule` | The correct answer depends on a condition stated in the policy, not the question |

The public benchmark sample exists for breadth and external comparability — a reader can check this system's general closed-book QA competence against a well-known floor, not just our own bespoke test. It's a **frozen, versioned snapshot** (stratified by source and category, seed `42`), not a live pull — reproducing it exactly requires a local clone of `genai_capability_bench` (its dataset files aren't part of the pip package); see the generation note in [`scenarios/fixtures/public_benchmark_sample.jsonl`](../scenarios/fixtures/public_benchmark_sample.jsonl)'s provenance metadata.

## Examples & sample results

Full report with charts: [`docs/samples/intended_performance_report.html`](samples/intended_performance_report.html) (open in a browser — GitHub shows raw HTML source, not the rendered page). Most recent run — Azure OpenAI, `gpt-5-5-20260424-gs`, API version `2025-04-01-preview`:

| Sub-dataset | n | avg score | pass rate |
|---|---|---|---|
| Custom golden set | 10 | 0.887 | 80.0% |
| Public benchmark sample | 30 | 0.860 | 86.7% |
| **Overall** | **40** | **0.87** | **85.0%** |

**6 tasks scored below threshold; the judge review split cleanly by sub-dataset, not randomly:**

- **2/2 custom-golden-set sub-threshold cases were scoring-metric artifacts, not failures.** e.g. `ip-08`: *"Yes. The flight is reimbursable because the portal outage was confirmed by IT."* — correct, but scored 0.68 because the model's phrasing didn't lexically overlap enough with the golden set's shorter reference answers.
- **4/4 public-benchmark sub-threshold cases were confirmed genuinely wrong.** Two were real wrong picks on ARC science items — one worth a caveat of its own: *"which substance retains the most energy from the Sun"* (gold answer "sand") is scientifically debatable on specific-heat-capacity grounds, a reminder that a public benchmark's gold answer isn't automatically ground truth either.

**A recurring technical finding, confirmed across three separate runs, not a one-off:** 1–2 tasks per run return an empty completion (no visible output at all, not an error) on the public benchmark sample specifically. The working hypothesis is that the 300-token `max_completion_tokens` budget is tight enough for this reasoning-family model to exhaust it on internal reasoning before producing a visible answer. This is now the **top next step** for this scenario — increase the token budget and confirm the empty completions stop.

**A finding that only shows up by re-running this notebook, not from a single pass:** an earlier run of this exact scenario against a different model in the gpt-5 family produced a materially different pass rate on the identical golden set. That's not a bug in this scenario's harness — it's a live instance of [Tier 2 — drift detection](../README.md#scenario-library) surfacing inside a Tier 1 test, and a concrete reason results here should be re-read after any model change, not assumed to carry over.

**Next steps for this scenario:** increase `max_completion_tokens` in both `configs/intended_performance*.yaml` and confirm the empty-completion cases stop recurring; wire `llm_judge_correctness` in as a proper secondary metric in `genai_capability_bench` rather than the ad hoc post-hoc pass used here; route any judge/human disagreement to a defined adjudication step.
