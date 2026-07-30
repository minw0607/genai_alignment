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

This scenario's first version used three ad hoc statistical choices that didn't hold up to scrutiny — see [Sample Results](#sample-results) for exactly what prompted each fix. All three are now named, cited methods from the published consistency/hallucination-detection literature, not house inventions.

**Chatbot consistency** is checked three ways, none of which is exact string matching:

- **Pass/fail flip rate** across 5 runs — inherits scenario 1's scoring (lexical/semantic match against a reference), so this is already tolerant of paraphrasing at the pass/fail level. A purely descriptive, uncorrected signal.
- **Semantic consistency** — the 5 raw answers are clustered by **bidirectional entailment** (does answer A imply answer B, and does B imply A?), and the score reported is `1 − normalized Shannon entropy` over the resulting cluster sizes. This is the **semantic entropy** method (Kuhn, Gal & Farquhar, 2024, *Nature*), adapted here to check entailment via an LLM-judge prompt rather than a dedicated NLI model (e.g. DeBERTa-MNLI, as the original paper uses), so it reuses this repo's existing target-model infrastructure instead of adding a new ML dependency. **This replaces an earlier version's TF-IDF cosine similarity against an unvalidated 0.5 threshold** — TF-IDF measures vocabulary overlap, not meaning, so a correct answer phrased differently each time could score low under it; the old function (`pairwise_self_consistency`) is kept in `reporting/repeat_run.py` as a free, zero-API-call fallback, but is no longer what this scenario calls by default.
- **BH-corrected reliability significance** — a one-sided exact binomial test per task (H0: true pass rate ≥ 80%, a stated policy floor, not something inferred from data), corrected across all 40 simultaneous tests via **Benjamini-Hochberg** FDR correction (Benjamini & Hochberg, 1995). This is the rigorous version of the raw flip-rate question: testing 40 tasks at once means some will look "significant" by chance alone, and this controls for that rather than reading each task's result in isolation. Pass-rate confidence itself is reported via a **Wilson score interval** (Wilson, 1927), not a normal-approximation standard error, since the normal approximation is poorly calibrated at n=5 (it can even produce an interval outside [0, 1]).

**Agentic consistency** is deliberately *not* text comparison of what the agent did — comparing raw trajectories as strings would penalize an agent for using different-but-equivalent phrasing in its reasoning, which isn't the failure mode this scenario cares about. Instead:
- **Task-outcome flip rate**, and its BH-corrected significance version — same methods as the chatbot track, applied to `task_passed`. With only 5 tasks per mode, the correction has little statistical power; treat it as a much weaker signal here than in the 40-task chatbot table.
- **Tool-selection variance** (`tool_f1` std dev across runs) — `tool_f1` is already equivalence-aware *upstream*, via `multi_agent_otel_eval`'s own flexible tool-equivalence mapping (calling a differently-named-but-functionally-equivalent tool isn't penalized as a wrong choice). High variance here means the agent's actions genuinely differed run to run, not that two runs used different words for the same action — though see Sample Results for what this metric actually showed in practice, which wasn't what was expected.

**A limitation stated plainly, not glossed over:** this scenario does **not** cleanly separate "variance from the model" and "variance from the orchestration layer wrapped around it." The chatbot track (one LLM call per task) is close to a variance floor; the agentic track (a multi-step loop, each step re-invoking the model) sits on top of that floor with compounded variance from planning, tool selection, and retries. Comparing the two tracks is a directional read on what orchestration adds, not a controlled ablation — a true decomposition would require running the same harness with orchestration switched on and off, which neither sibling repo supports today. That's flagged as a next step, not solved here.

**A second limitation, also stated plainly:** the bidirectional-entailment check above uses this repo's own target model as the judge, the same same-model-family caveat noted in scenario 1's judge review. A genuinely independent check (a dedicated NLI model, or a different model family) is a documented next step, not something this version claims to have solved.

## Data

| Track | Data | Scale |
|---|---|---|
| Chatbot | Scenario 1's two golden sets, unchanged | 40 tasks × 5 repeats = 200 calls |
| Agentic | Fixed Mind2Web task subset (NeurIPS 2023 web-navigation benchmark), via `multi_agent_otel_eval`'s cached loader | 5 tasks × 5 repeats × 2 modes = 50 agent runs |

No new data was authored for this scenario — it reuses scenario 1's golden sets for the chatbot track, and a public benchmark (Mind2Web) already integrated into the agentic sibling repo for the other. The point of this scenario is the *repeat* dimension, not new content.

## Sample Results

Full report with charts: [`docs/samples/consistency_reliability_report.html`](samples/consistency_reliability_report.html) (open in a browser — GitHub shows raw HTML source, not the rendered page). Azure OpenAI, `gpt-5-5-20260424-gs`, API version `2025-04-01-preview` — this run, with the statistically-refined methodology:

| Track | Raw flip rate | BH-corrected significant | Headline finding |
|---|---|---|---|
| Chatbot (40 tasks × 5 repeats) | 7/40 (18%) | **3/40** | 2 tasks passed every run but split into >1 meaning-cluster |
| Agentic (5 tasks × 5 repeats × 2 modes) | 0/10 (0%) | 0/10 | `tool_f1_std` ≈ 0.01 — see below |

**The correction did real work, not just cosmetic rigor:** 7 tasks showed *some* pass/fail disagreement across the 5 runs, but after Benjamini-Hochberg correction for testing 40 tasks simultaneously (against an 80% reliability floor), only **3** are statistically distinguishable from acceptable reliability — the other 4 raw flips are not strong enough evidence to act on once you account for testing this many tasks at once. This is the concrete difference between "did anything disagree" and "is this actually a finding," and it's the reason this refinement was worth doing rather than a theoretical nicety.

**Semantic consistency caught something pass/fail entirely missed:** 2 tasks passed every one of their 5 runs, yet their answers still split into more than one bidirectional-entailment meaning-cluster — the model reached a passing score consistently, but didn't always mean the same thing when it did. A pure pass/fail view would have reported these as perfectly fine.

**Agentic track — the same surprising `tool_f1` finding recurred:** average `tool_f1_std` came back at 0.01 — essentially flat — while `n_tool_calls` varied substantially, the same pattern observed in earlier runs. This framework's Tool Correctness scoring appears to be driven by whether the required actions happened *at least once*, not by how much extra exploration occurred around them; `n_tool_calls_std` continues to carry more of the real behavioral-variance signal than `tool_f1_std` does. Neither track flipped significantly after BH correction this run — though with only 5 tasks per mode, that correction has very little statistical power, so a clean result here is weaker evidence than the same result on the 40-task chatbot table.

**Model-vs-harness variance, stated honestly:** the chatbot track (one LLM call per task) is close to a variance floor; the agentic track's variance sits on top of that floor, compounded across a multi-step orchestration loop. Comparing the two tracks is a directional read on what orchestration adds, not a controlled ablation — that would need the same harness run with orchestration on and off, which neither sibling repo supports today. Flagged as the top next step, not glossed over.

**A process note worth keeping:** the first full run of this notebook crashed partway through — `TfidfVectorizer`'s default tokenizer rejects single-character answers (a bare `"5"` has no 2+-character tokens to vectorize), which `pairwise_self_consistency` didn't handle. Fixed in [`reporting/repeat_run.py`](../reporting/repeat_run.py) with a looser token pattern and an exact-match fallback. The crash happened before the (much more expensive) agentic track started, so the re-run only had to redo the cheaper chatbot calls.

**Next steps for this scenario:** build a controlled ablation (same task, orchestration on vs. off) to actually isolate how much variance the harness adds on top of the model's own; replace the LLM-judge entailment check with a dedicated NLI model for a genuinely independent semantic-consistency signal; revisit the 80% reliability floor once there's a real SLA or business requirement to calibrate it against rather than a stated placeholder.

## References

- Kuhn, L., Gal, Y., & Farquhar, S. (2024). [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0). *Nature*. — bidirectional-entailment clustering + Shannon entropy; the basis for this scenario's semantic-consistency metric.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*. — the confidence-interval method used for pass-rate estimates.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*. — the multiple-comparison correction applied before flagging any task as significantly unreliable.
- Manakul, P., Liusie, A., & Gales, M. (2023). [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896). — the "sample N times and check mutual support" pattern this scenario's chatbot track is built on; its five comparison methods (BERTScore, MQAG, n-gram, NLI, LLM-prompting) are the direct precedent for using an entailment-style check instead of lexical similarity.
- Wang, X., Wei, J., Schuurmans, D., et al. (2022). [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171). — origin of the "self-consistency" term; an inference-time technique (majority-vote over reasoning paths), distinct from the measurement problem this scenario addresses, but the source of the shared vocabulary.
- Cavalin, P., Sanctos, C., Grave, M., Pinhanez, C., & Primerano, Y. (2025). [CAT: A Metric-Driven Framework for Analyzing the Consistency-Accuracy Relation of LLMs under Controlled Input Variations](https://arxiv.org/abs/2512.23711). — proposes reporting accuracy and consistency jointly (Consistency-Accuracy-Relation curves, the CORE index) rather than as two separate numbers; a candidate direction for this scenario's reporting, not yet adopted.
- NIST AI Risk Management Framework and ISO/IEC 42001 — both name reliability (including repeatability and resistance to drift) as a core trustworthy-AI characteristic, confirming this scenario tests a recognized governance dimension; neither prescribes a specific statistical method, which is why the choices above are drawn from the academic literature instead.
- [`alepot55/agentrial`](https://github.com/alepot55/agentrial) — an open-source framework built around the same "run an agent N times, compute confidence intervals, correct for multiple comparisons" premise as this scenario's agentic track. Evaluated as a possible adoption for the agentic track specifically; its core statistical methods (Wilson intervals, BH correction) are implemented natively here instead, since adopting the framework wholesale would duplicate rather than add to what's already built.
