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

Both tracks share one generic base function — [`reporting/repeat_run.py`](../reporting/repeat_run.py)'s `variance_by_task` — plus two statistical layers on top (Wilson confidence intervals, Benjamini-Hochberg-corrected significance testing), configured with different column names rather than duplicated. That module is deliberately not scenario-specific: drift detection ([`docs/drift_detection.md`](drift_detection.md)) needs the same "run N times, compare" mechanic for its baseline-capture step, so the repeat-run harness is built once, here, for both to use.

**A setup difference from scenario 1 worth flagging:** `multi_agent_otel_eval` has no `pyproject.toml`, so it isn't pip-installable like the other sibling repos. The adapter references a local sibling clone via `sys.path` instead — see [README — Setup](../README.md#setup). Because this scenario's setup surface is larger than scenario 1's (a sibling clone *and* a separate dependency extra), the notebook's environment-check cell explicitly verifies the sibling repo is present before anything else runs — see [`reporting/env_check.py`](../reporting/env_check.py).

**This scenario also closes a persistence gap scenario 1 didn't have to deal with.** `genai_capability_bench`'s `run_from_config` writes each run to a fixed output directory keyed by `run_id` — fine for a single run, but calling it in a loop for repeats overwrites the same directory each time, so only the *last* of 5 repeats would survive on disk. `scenario.save_artifacts()` persists the full combined raw results (all repeats, both tracks) to their own files specifically so the documentation trail has real evidence behind the aggregated numbers, not just the last repeat.

## Methodology

Three statistical methods, each a named, cited technique from the consistency/hallucination-detection literature rather than a house invention:

| Signal | Method | Why |
|---|---|---|
| Text self-consistency | **Semantic consistency** — cluster the N raw answers by bidirectional entailment (does A imply B *and* B imply A?), score `1 − normalized Shannon entropy` over cluster sizes | The **semantic entropy** method (Kuhn, Gal & Farquhar, 2024, *Nature*), adapted to check entailment via an LLM-judge prompt (`JUDGE_MODEL`, a different deployment than the target model) rather than a dedicated NLI model (e.g. DeBERTa-MNLI, as the original paper uses) — reuses this repo's existing model-client infrastructure instead of adding a new ML dependency. |
| Pass-rate confidence | **Wilson score interval** (Wilson, 1927) | Well-calibrated at small n (n=5 here), unlike a normal-approximation standard error, which can even produce an interval outside [0, 1]. |
| Which tasks are unreliable | **Benjamini-Hochberg-corrected significance test** (Benjamini & Hochberg, 1995) against a stated 80% reliability floor | A one-sided exact binomial test per task, corrected across all simultaneous tests — controls the false-discovery rate rather than reading each task's raw `flips` flag in isolation. |

**One more layer on top of that significance test:** `significant_below_floor` alone conflates two different findings — a task that genuinely flips run to run, and a task that fails (or passes) *every single run identically*, which isn't instability at all. `add_reliability_category` (`reporting/repeat_run.py`) splits this into three labels: `unstable` (significant and flips — a real consistency finding), `consistently_failing` (significant, never flips — a capability gap or scoring-threshold artifact riding along on the same test, not instability), and `reliable`. This was a real fix, not a hypothetical one: reviewing an early run of this scenario found 5 of 6 "significantly unreliable" chatbot tasks were `consistently_failing`, not `unstable` — 3 were the same ARC science items scenario 1 already found the model consistently gets wrong (semantic_consistency ≈ 1.0 — the model confidently repeats the same incorrect answer), and 2 were the same custom-golden-set tasks scenario 1's judge already confirmed as correct-but-under-scored. Only 1 of the 6 was genuine instability.

**Chatbot consistency** uses all three: pass/fail flip rate (a raw, descriptive signal — inherits scenario 1's scoring, so already tolerant of paraphrasing at the pass/fail level), semantic consistency on the raw answers, and BH-corrected significance on top of the pass rate. **Agentic consistency** is deliberately *not* text comparison of what the agent did — comparing raw trajectories as strings would penalize an agent for using different-but-equivalent phrasing, which isn't the failure mode this scenario cares about. Instead: task-outcome flip rate (plus its BH-corrected version — with only 5 tasks per mode, the correction has little statistical power there, a much weaker signal than the 40-task chatbot table), and tool-selection variance (`tool_f1` std dev), which is already equivalence-aware *upstream* via `multi_agent_otel_eval`'s own flexible tool-equivalence mapping — though see Sample Results for what this metric actually showed in practice, which wasn't what was expected.

**Both statistical rigor and cost are configurable, not fixed.** `scenario.chatbot_variance(..., use_semantic_entropy=False)` swaps semantic consistency for a free, zero-API-call TF-IDF check (`reporting.repeat_run.add_pairwise_consistency` / `pairwise_self_consistency`) — same output columns, no client required, faster for dev iteration or a cost-constrained run, at the cost of the accuracy semantic consistency was built to fix (see Sample Results for the specific cases it caught that TF-IDF didn't). The reliability floor (80%) and significance level (α=0.05) are both named constants in `scenarios/consistency_reliability.py`, stated policy choices rather than derived from data.

**A limitation stated plainly, not glossed over:** this scenario does **not** cleanly separate "variance from the model" and "variance from the orchestration layer wrapped around it." The chatbot track (one LLM call per task) is close to a variance floor; the agentic track (a multi-step loop, each step re-invoking the model) sits on top of that floor with compounded variance from planning, tool selection, and retries. Comparing the two tracks is a directional read on what orchestration adds, not a controlled ablation — a true decomposition would require running the same harness with orchestration switched on and off, which neither sibling repo supports today. That's flagged as a next step, not solved here.

**A second limitation, stated plainly:** the bidirectional-entailment check is an LLM-judge prompt, not a dedicated NLI model — `JUDGE_MODEL` is a genuinely different deployment from the target model (removing the same-model-family caveat this scenario used to carry), but it's still an LLM doing the judging, with whatever judge-side variance that implies. A dedicated NLI model (e.g. DeBERTa-MNLI, as Kuhn et al. 2024 use) would remove that remaining source of variance, at the cost of a new ML dependency — a documented next step, not something this version claims to have solved.

## Data

| Track | Data | Scale |
|---|---|---|
| Chatbot | Scenario 1's two golden sets, unchanged | 40 tasks × 5 repeats = 200 calls |
| Agentic | Fixed Mind2Web task subset (NeurIPS 2023 web-navigation benchmark), via `multi_agent_otel_eval`'s cached loader | 5 tasks × 5 repeats × 2 modes = 50 agent runs |

No new data was authored for this scenario — it reuses scenario 1's golden sets for the chatbot track, and a public benchmark (Mind2Web) already integrated into the agentic sibling repo for the other. The point of this scenario is the *repeat* dimension, not new content.

## Sample Results

Full report with charts: [`docs/samples/consistency_reliability_report.html`](samples/consistency_reliability_report.html) (open in a browser — GitHub shows raw HTML source, not the rendered page). Azure OpenAI target `gpt-5-5-20260424-gs`, judged by a genuinely independent deployment (`JUDGE_MODEL=gpt-5-6-terra-latest-gs`), API version `2025-04-01-preview`:

| Track | Raw flip rate | Genuinely unstable | Consistently failing | Headline finding |
|---|---|---|---|---|
| Chatbot (40 tasks × 5 repeats) | 4/40 (10%) | **0/40** | **5/40** | Zero real instability this run — all 5 flagged tasks fail identically every time |
| Agentic (5 tasks × 5 repeats × 2 modes) | 7/10 (70%) | **2/10** | **2/10** | Both categories non-zero for the first time — see below |

**Switching to a genuinely independent judge (this run) materially changed the agentic picture.** Every prior run of this scenario used an unintentional same-model-family-adjacent judge (a leftover `.env` default, not a deliberate choice) and reported 0/10 unstable and 0/10 consistently failing on the agentic track every time. With `JUDGE_MODEL` now a deliberately different, independent deployment, the agentic track shows real signal in both categories for the first time: 2 tasks genuinely unstable, 2 consistently failing. This is the independent-judge change doing its job — the previous judge was, in effect, too lenient or too correlated with the target model to surface this.

**A reproducibility caveat, observed directly, not theoretical:** re-running the agentic track twice in a row with the identical independent judge produced two different specific pictures — one run classified all 5 single-agent tasks as `consistently_failing` and all 5 multi-agent tasks as `reliable`; the very next run (reported above) spread `unstable`/`consistently_failing` across both modes instead. The chatbot track's classification was stable across the same two runs; the agentic track's was not. This is exactly the low-statistical-power caveat this doc already flags for BH correction at n=5 — restated here because it was actually observed, not just theorized. **Practical takeaway: at 5 repeats per agentic task, treat which specific tasks land in which category as noisy; treat "the agentic track now shows real unstable/consistently-failing signal under an independent judge, where it showed none under the old one" as the stable finding.** Increasing agentic repeats beyond 5 is the concrete next step this observation motivates.

**Chatbot track — the categorization itself held stable, only the count shifted slightly:** 0/40 unstable in every run so far; consistently-failing count moved from 3 to 5 between runs (avg semantic consistency 0.96 this run). Checking `semantic_consistency` on the 5 flagged rows (all ≈1.0 — the model confidently repeats the same answer) confirms these are the same kind of ARC-science and custom-golden-set items scenario 1 already characterized: some genuine capability gaps, some verbose-but-correct answers scenario 1's judge already confirmed. Not new information from this scenario — scenario 1 findings that would have been miscounted as consistency findings without the `reliability_category` split.

**Semantic consistency caught something pass/fail entirely missed:** 2 tasks passed every one of their 5 runs, yet their answers still split into more than one bidirectional-entailment meaning-cluster — the model reached a passing score consistently, but didn't always mean the same thing when it did. A pure pass/fail view would have reported these as perfectly fine.

**Agentic `tool_f1` — the same flat-variance finding recurred:** average `tool_f1_std` came back essentially zero (≈0.00) across both modes even where task outcomes flipped, while `n_tool_calls` varied more — the same pattern observed in every run so far. This framework's Tool Correctness scoring appears to be driven by whether the required actions happened *at least once*, not by how much extra exploration occurred around them.

**Model-vs-harness variance, stated honestly:** the chatbot track (one LLM call per task) is close to a variance floor; the agentic track's variance sits on top of that floor, compounded across a multi-step orchestration loop. Comparing the two tracks is a directional read on what orchestration adds, not a controlled ablation — that would need the same harness run with orchestration on and off, which neither sibling repo supports today. Flagged as the top next step, not glossed over.

**A process note worth keeping:** the first full run of this notebook crashed partway through — `TfidfVectorizer`'s default tokenizer rejects single-character answers (a bare `"5"` has no 2+-character tokens to vectorize), which `pairwise_self_consistency` didn't handle. Fixed in [`reporting/repeat_run.py`](../reporting/repeat_run.py) with a looser token pattern and an exact-match fallback. The crash happened before the (much more expensive) agentic track started, so the re-run only had to redo the cheaper chatbot calls.

**Next steps for this scenario:** increase agentic repeats beyond 5 — two independent runs under the same independent judge classified different specific tasks as `unstable` vs. `consistently_failing`, direct evidence the current n is too small for stable per-task categorization there; build a controlled ablation (same task, orchestration on vs. off) to actually isolate how much variance the harness adds on top of the model's own; replace the LLM-judge entailment check with a dedicated NLI model for a genuinely independent semantic-consistency signal; revisit the 80% reliability floor once there's a real SLA or business requirement to calibrate it against rather than a stated placeholder.

## References

- Kuhn, L., Gal, Y., & Farquhar, S. (2024). [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0). *Nature*. — bidirectional-entailment clustering + Shannon entropy; the basis for this scenario's semantic-consistency metric.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*. — the confidence-interval method used for pass-rate estimates.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*. — the multiple-comparison correction applied before flagging any task as significantly unreliable.
- Manakul, P., Liusie, A., & Gales, M. (2023). [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896). — the "sample N times and check mutual support" pattern this scenario's chatbot track is built on; its five comparison methods (BERTScore, MQAG, n-gram, NLI, LLM-prompting) are the direct precedent for using an entailment-style check instead of lexical similarity.
- Wang, X., Wei, J., Schuurmans, D., et al. (2022). [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171). — origin of the "self-consistency" term; an inference-time technique (majority-vote over reasoning paths), distinct from the measurement problem this scenario addresses, but the source of the shared vocabulary.
- Cavalin, P., Sanctos, C., Grave, M., Pinhanez, C., & Primerano, Y. (2025). [CAT: A Metric-Driven Framework for Analyzing the Consistency-Accuracy Relation of LLMs under Controlled Input Variations](https://arxiv.org/abs/2512.23711). — proposes reporting accuracy and consistency jointly (Consistency-Accuracy-Relation curves, the CORE index) rather than as two separate numbers; a candidate direction for this scenario's reporting, not yet adopted.
- NIST AI Risk Management Framework and ISO/IEC 42001 — both name reliability (including repeatability and resistance to drift) as a core trustworthy-AI characteristic, confirming this scenario tests a recognized governance dimension; neither prescribes a specific statistical method, which is why the choices above are drawn from the academic literature instead.
- [`alepot55/agentrial`](https://github.com/alepot55/agentrial) — an open-source framework built around the same "run an agent N times, compute confidence intervals, correct for multiple comparisons" premise as this scenario's agentic track. Evaluated as a possible adoption for the agentic track specifically; its core statistical methods (Wilson intervals, BH correction) are implemented natively here instead, since adopting the framework wholesale would duplicate rather than add to what's already built.
