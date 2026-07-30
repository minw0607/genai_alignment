# Consistency & Reliability — Scenario Design

[← Back to README](../README.md)

**Tier 1 — Foundational behavior.** Chatbot track is a **reused adapter** (scenario 1's `genai_capability_bench` adapter, unchanged, run repeatedly). Agentic track is a **new adapter** onto [`multi_agent_otel_eval`](https://github.com/minw0607/multi_agent_otel_eval). Notebook: [`notebooks/02_consistency_reliability.ipynb`](../notebooks/02_consistency_reliability.ipynb) · Sample report: [`docs/samples/consistency_reliability_report.html`](samples/consistency_reliability_report.html).

---

## Scope

Whether the system gives essentially the same answer, or takes essentially the same action, when asked the same thing twice. "Essentially the same" is doing real work in that sentence — this scenario exists because neither a chatbot's wording nor an agent's exact tool-call sequence needs to be byte-identical to be consistent, and testing for consistency only works if the test can tell the difference between "phrased differently" and "actually different."

| | |
|---|---|
| **Risk** | Same input yields different output; hallucination / variance. |
| **Goal** | Repeatable, consistent outputs for equivalent inputs. |

Scenario 1 surfaced this by accident, not by design: re-running its exact golden set against the same model on separate occasions produced different pass rates. This scenario turns that into the actual test rather than a footnote.

## Approach

Two tracks, because "consistent" means something different depending on what the system under test actually does:

- **Chatbot track** — scenario 1's exact two golden sets ([`scenarios/fixtures/intended_performance.jsonl`](../scenarios/fixtures/intended_performance.jsonl), [`scenarios/fixtures/public_benchmark_sample.jsonl`](../scenarios/fixtures/public_benchmark_sample.jsonl)), run through the unchanged `adapters/capability_bench.py` adapter, repeated 5 times each.
- **Agentic track** — a new [`adapters/agent_otel.py`](../adapters/agent_otel.py) wrapping `multi_agent_otel_eval`'s `evaluate_batch()`, run on a fixed 5-task Mind2Web subset, 5 times each, in both single-agent (ReAct) and multi-agent (planner/navigator/validator) mode.

Both tracks share one generic function — [`reporting/repeat_run.py`](../reporting/repeat_run.py)'s `variance_by_task` — configured with different column names rather than duplicated. That module is deliberately not scenario-specific: drift detection ([`docs/drift_detection.md`](drift_detection.md)) needs the same "run N times, compare" mechanic for its baseline-capture step, so the repeat-run harness is built once, here, for both to use.

**A setup difference from scenario 1 worth flagging:** `multi_agent_otel_eval` has no `pyproject.toml`, so it isn't pip-installable like the other sibling repos. The adapter references a local sibling clone via `sys.path` instead — see [README — Setup](../README.md#setup). Because this scenario's setup surface is larger than scenario 1's (a sibling clone *and* a separate dependency extra), the notebook's environment-check cell explicitly verifies the sibling repo is present before anything else runs — see [`reporting/env_check.py`](../reporting/env_check.py).

**This scenario also closes a persistence gap scenario 1 didn't have to deal with.** `genai_capability_bench`'s `run_from_config` writes each run to a fixed output directory keyed by `run_id` — fine for a single run, but calling it in a loop for repeats overwrites the same directory each time, so only the *last* of 5 repeats would survive on disk. `scenario.save_artifacts()` persists the full combined raw results (all repeats, both tracks) to their own files specifically so the documentation trail has real evidence behind the aggregated numbers, not just the last repeat.

## Methodology

**Chatbot consistency** is checked two ways, neither of which is exact string matching:
- **Pass/fail flip rate** across 5 runs — inherits scenario 1's scoring (lexical/semantic match against a reference), so this is already tolerant of paraphrasing at the pass/fail level.
- **Self-consistency** — average pairwise TF-IDF cosine similarity among the 5 raw answers *to each other* (not against the reference). This exists because pass/fail alone can hide instability: a task can pass every run while phrasing the answer completely differently each time, which pass-rate consistency wouldn't catch.

**Agentic consistency** is deliberately *not* text comparison of what the agent did — comparing raw trajectories as strings would penalize an agent for using different-but-equivalent phrasing in its reasoning, which isn't the failure mode this scenario cares about. Instead:
- **Task-outcome flip rate** — same idea as the chatbot track, applied to `task_passed`.
- **Tool-selection variance** (`tool_f1` std dev across runs) — `tool_f1` is already equivalence-aware *upstream*, via `multi_agent_otel_eval`'s own flexible tool-equivalence mapping (calling a differently-named-but-functionally-equivalent tool isn't penalized as a wrong choice). High variance here means the agent's actions genuinely differed run to run, not that two runs used different words for the same action.

**A limitation stated plainly, not glossed over:** this scenario does **not** cleanly separate "variance from the model" and "variance from the orchestration layer wrapped around it." The chatbot track (one LLM call per task) is close to a variance floor; the agentic track (a multi-step loop, each step re-invoking the model) sits on top of that floor with compounded variance from planning, tool selection, and retries. Comparing the two tracks is a directional read on what orchestration adds, not a controlled ablation — a true decomposition would require running the same harness with orchestration switched on and off, which neither sibling repo supports today. That's flagged as a next step, not solved here.

## Data

| Track | Data | Scale |
|---|---|---|
| Chatbot | Scenario 1's two golden sets, unchanged | 40 tasks × 5 repeats = 200 calls |
| Agentic | Fixed Mind2Web task subset (NeurIPS 2023 web-navigation benchmark), via `multi_agent_otel_eval`'s cached loader | 5 tasks × 5 repeats × 2 modes = 50 agent runs |

No new data was authored for this scenario — it reuses scenario 1's golden sets for the chatbot track, and a public benchmark (Mind2Web) already integrated into the agentic sibling repo for the other. The point of this scenario is the *repeat* dimension, not new content.

## Sample Results

Full report with charts: [`docs/samples/consistency_reliability_report.html`](samples/consistency_reliability_report.html) (open in a browser — GitHub shows raw HTML source, not the rendered page). Azure OpenAI, `gpt-5-5-20260424-gs`, API version `2025-04-01-preview`:

| Track | Flip rate | Headline variance signal |
|---|---|---|
| Chatbot (40 tasks × 5 repeats) | 5/40 (12%) flipped pass/fail | Self-consistency: 1 task scored < 0.5 despite mostly passing |
| Agentic (5 tasks × 5 repeats × 2 modes) | 1/5 single-agent, 0/5 multi-agent | See below — not `tool_f1_std` |

**Chatbot track:** 5 of 40 tasks flipped pass/fail across identical repeated runs — with temperature omitted for this reasoning-family model, so this isn't coming from an explicit sampling knob. One task (`mmlu_formal_logic_2454`) had both a low pass rate *and* low self-consistency (0.0) — genuinely unstable, not just borderline. One other task passed consistently but scored self-consistency well below 1.0 — correct every time, worded differently enough each time that a naive text-diff would flag it as unstable, which is exactly why self-consistency is tracked as a separate signal from pass/fail rather than folded into it.

**Agentic track — a real finding, not the one this scenario was designed to surface:** `tool_f1` showed essentially **zero variance across every one of the 10 task/mode combinations**, even though `n_tool_calls` varied substantially (std up to 33 calls on the same task). Read plainly: this framework's Tool Correctness scoring appears to be driven by whether the required actions were performed *at least once*, not by how much extra exploration or redundant tool-calling happens around them. That means `tool_f1_std` — the metric this scenario was built around as the "consistency of actions" signal — turned out not to be sensitive to the kind of variance actually present in these runs; `n_tool_calls_std` carried the real signal instead (multi-agent averaged 22.0, single-agent 15.4). This is exactly the kind of thing a first real run is supposed to surface, and it's now reflected in `scenarios/consistency_reliability.py`'s observation logic — it checks for a flat `tool_f1_std` and reports the more informative metric when that happens, rather than always repeating a boilerplate sentence about a number that turned out to be uninformative.

**Single-agent flipped on one task (task 2, `discogs`, pass rate 0.6); multi-agent didn't flip on any of the 5** — a small, single-run data point in favor of the multi-agent system's own stated value proposition (more consistent completion, per `multi_agent_otel_eval`'s own README), not yet enough repeats to call it a settled finding.

**Model-vs-harness variance, stated honestly:** the chatbot track (one LLM call per task) is close to a variance floor; the agentic track's variance sits on top of that floor, compounded across a multi-step orchestration loop. Comparing the two tracks is a directional read on what orchestration adds, not a controlled ablation — that would need the same harness run with orchestration on and off, which neither sibling repo supports today. Flagged as the top next step, not glossed over.

**A process note worth keeping:** the first full run of this notebook crashed partway through — `TfidfVectorizer`'s default tokenizer rejects single-character answers (a bare `"5"` has no 2+-character tokens to vectorize), which `pairwise_self_consistency` didn't handle. Fixed in [`reporting/repeat_run.py`](../reporting/repeat_run.py) with a looser token pattern and an exact-match fallback. The crash happened before the (much more expensive) agentic track started, so the re-run only had to redo the cheaper chatbot calls.

**Next steps for this scenario:** build a controlled ablation (same task, orchestration on vs. off) to actually isolate how much variance the harness adds on top of the model's own; extend self-consistency scoring to embedding-based similarity, which will under-count paraphrases using very different vocabulary than TF-IDF can see.
