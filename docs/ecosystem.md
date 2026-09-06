# Where This Repo Sits — and What It Borrows

[← Back to README](../README.md)

A short positioning note: what the public evaluation ecosystem provides, what this repo provides, and which pieces are worth taking rather than rebuilding.

Written because "why not just use an existing eval framework?" is a fair question with a specific answer, and because two of these tools are planned dependencies rather than alternatives.

---

## The landscape has three layers. This repo is in none of them.

| Layer | Representative tools | The question it answers |
|---|---|---|
| **Capability benchmarks + harnesses** | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [HELM](https://crfm.stanford.edu/helm/), OpenCompass, [Inspect AI](https://inspect.aisi.org.uk/), [GDPval](https://arxiv.org/abs/2510.04374) | How capable is this model? |
| **Application metric frameworks** | [RAGAS](https://github.com/explodinggradients/ragas), [DeepEval](https://github.com/confident-ai/deepeval), [Promptfoo](https://www.promptfoo.dev/) | Does this application hit its metrics? |
| **Hosted observability + eval** | [LangSmith](https://www.langchain.com/langsmith), Braintrust, Phoenix / Arize | What happened in production, and did it regress? |

This repo is a **scenario-design layer**: contrast ladders, floor arms, honest denominators, case-level statistics with a reported power floor, and utility measured beside safety. That is methodology, not machinery.

No tool above supplies it, and none of them prevents you from getting it wrong. A framework will happily run an underpowered comparison, average a judge score with a deterministic one, or report a compliance rate whose denominator quietly excludes the runs that failed.

**The corollary matters for scope discipline:** everything useful in the ecosystem sits either *upstream* of this repo (generating cases) or *downstream* (running and tracking them). The middle has no off-the-shelf equivalent, which is the part worth building.

---

## Assessments

### Bloom — Anthropic · **adopted, for generation only** — [method note](fixture_generation.md)

[Bloom](https://alignment.anthropic.com/2025/bloom-auto-evals/) (December 2025) takes a single researcher-specified behavior and automatically produces a whole evaluation suite through a four-stage pipeline — understand, ideate, roll out, judge. Anthropic built four alignment suites with it (delusional sycophancy, instructed long-horizon sabotage, self-preservation, self-preferential bias), each 100 rollouts × 3 repeats across 16 frontier models, and validated the output against hand-labelled judgments.

**Why it fits.** Every scenario here is already a behavior specification — *does it disclose beyond policy*, *does it honour a stated limit*, *does a record survive a handoff*. Bloom is built to turn precisely that into hundreds of cases, which is the answer to this library's most persistent weakness: hand-authored fixtures at n = 6, n = 9, n = 15, where several findings have landed on *"underpowered, not null"* rather than on a result.

**What is deliberately not taken.** Bloom's metric is an LLM judge — elicitation rate, the share of rollouts scoring ≥ 7/10 for behavior presence. Tier 2 and Tier 3 scenarios here keep a judge out of the primary path on purpose. Generated cases are scored by this repo's own deterministic scorers, against its own floor arms.

**What was built.** The loop lives in [`generation/`](../generation/) — behaviour spec, ideate from the hand-authored seeds, then mechanical gates, then a second-model quality check whose verdicts are recorded as provenance and gate nothing. Piloted on [Multi-Agent Handoff](multi_agent_handoff.md), which went from **6 records to 30**: 40 proposals, 9 rejected by the gates, every rejection on a profile boundary. Full method, including the two criteria the quality judge invented for itself, in [Growing a Fixture Without Losing It](fixture_generation.md).

**And one risk that comes with it.** Generation at scale reintroduces author bias — cases produced by the same process that scores them. Bloom without an independent check turns a small-n problem into a large-n problem with the same blind spot. The pilot scenario minimises it by construction (its ground truth *is* the generated record, so there is no target to move); the general mitigation is the external reproduction track that [Tool / MCP Abuse](tool_mcp_abuse.md) already runs, generalised.

### Inspect AI — UK AI Security Institute · **export target, built** — [method note](interop_inspect.md)

[Inspect AI](https://inspect.aisi.org.uk/) is the only widely-used framework in the landscape built for *safety and alignment* evaluation rather than capability or application metrics. Task / solver / scorer model, pytest-native, runs locally, open source.

**What it buys: portability.** Nobody outside this repo could run these scenarios or compare against them, which is the standing weakness of use-case-grounded work.

**What it does not buy: methodology.** Inspect supplies a clean runtime. It does not supply the contrast ladder, the floor arm, honest denominators, or `min_attainable_pvalue`. Refactoring scenarios into its abstraction "for the methods" would cost a large migration and return nothing methodological — the design here is the more opinionated of the two. Treated as an **export format**, not a replacement.

**What was built.** [`interop/`](../interop/) exports a scenario's cases as an Inspect dataset, its results as one eval log per arm, and a task stub wiring the two — so an Inspect user can run *their* agent against *these* cases. Piloted on [Boundary / Permission](boundary_permission.md); the output round-trips through Inspect's own reader, and `inspect_ai` is deliberately not a dependency. Details, including the two schema errors that only surfaced when the logs were loaded back: [Exporting to Inspect AI](interop_inspect.md).

### GDPval — OpenAI · **not applicable, but its task discipline is worth reading**

[GDPval](https://arxiv.org/abs/2510.04374) ([announcement](https://openai.com/index/gdpval/)) is 1,320 tasks across 44 occupations in the nine largest GDP-contributing US sectors, authored by professionals averaging 14 years of experience, graded by blind head-to-head human comparison against a real expert's deliverable.

It is a capability benchmark of unusually high quality, and it answers the first of the two questions this repo explicitly does not ask. It has no notion of authorization, scope, disclosure, or resource limits, and its grading is human preference.

**Worth borrowing:** its construction discipline. Every task derives from an actual work product, with reference files and a defined deliverable. That is the same instinct behind use-case grounding here, executed at a scale hand-authoring cannot reach.

### OpenAI Evals — **not a fit**

[openai/evals](https://github.com/openai/evals) is a benchmark registry built around completion functions and a registry of eval definitions: *did the model produce the right answer?* The repository now directs users to the hosted Evals product in the OpenAI dashboard.

It answers the capability question. Nothing here to adopt.

### LangSmith — **relevant infrastructure, with a conflict worth naming**

[LangSmith](https://www.langchain.com/langsmith)'s traces → datasets → evals pipeline overlaps with what `reporting/` does by hand. The genuinely useful piece is **experiment tracking across runs**, which this repo lacks entirely and has been bitten by twice — see [cross-run tracking](#cross-run-tracking) below.

**The conflict.** LangSmith is hosted. This repo maintains deliberate confidentiality discipline — generic model labels, scrubbed provider strings, and a `scrub_provider_text` helper written specifically because raw API error text leaked a vendor name into a committed sample report. Shipping run data to a third-party platform cuts against that, and against the claim that every report is reproducible from committed artifacts.

The capability is worth having; the hosting is not worth the trade here.

### DeepEval · Promptfoo · RAGAS — **sibling-repo territory**

Pytest-style regression gating, prompt/provider matrix comparison with a large red-team vector set, and RAG-specific metrics respectively. Each maps onto a sibling repo's remit — [`llm_red_teaming`](https://github.com/minw0607/llm_red_teaming) and [`rag_eval_framework`](https://github.com/minw0607/rag_eval_framework) — more naturally than onto this one.

---

<a id="cross-run-tracking"></a>

## What is being built here instead

### Cross-run tracking — build, not buy

Every run currently **overwrites the last**. `outputs/runs/<scenario>/` is rewritten in place, and nothing records that a previous run existed or what it said.

That is how two real instabilities went unnoticed:

| Scenario | Metric | Earlier run | Later run |
|---|---|---|---|
| [Multi-Agent Handoff](multi_agent_handoff.md) | `small_relay` cases failing | 4/6 — *p* = 0.061, not significant | 6/6 — *p* = 0.002, **significant** |
| [Resource & Budget](resource_budget.md) | unbudgeted tool calls, same ticket | — | **3× spread** on identical input |

Same code, same fixtures, same model. The first crossed a significance threshold between runs and was caught by accident.

**Built** — `reporting/run_log.py`, wired into every scenario's `save_artifacts`:

- **Runs are kept, not overwritten.** Each run's artifacts are snapshotted to `outputs/runs/<scenario>/_archive/<run_id>/`. The flat files every notebook reads are still written exactly as before, so nothing that reads `raw_results.csv` had to change — the archive is a copy taken alongside them.
- **One row per run** in `outputs/runs/index.csv`: run id, scenario, git SHA, whether the tree was dirty, model, sample size, and a headline metric.
- **`compare_runs()`** answers the actual question — per scenario and headline, did the value move between runs, and what were all the values it took.

`git_dirty` is recorded because a run produced by uncommitted code is not reproducible from its SHA, and that is worth knowing when a number moves.

Regenerating a report from saved data is *not* a run: the archiver skips any output directory outside `outputs/runs/`, so the maintenance scripts that rebuild sample reports against a temp directory never appear in the ledger.

`outputs/` is gitignored, so archives stay on the machine that produced them — no raw model output enters version control.

### Why not a hosted platform for this

The repo's auditability claim rests on every report being regenerable from artifacts committed alongside it. A hosted dependency weakens that, and the confidentiality discipline above makes it worse rather than better. Twenty lines of local CSV buys the property that actually matters.

---

## Summary

| Tool | Verdict |
|---|---|
| **Bloom** | **Adopted** for case generation — [method note](fixture_generation.md); scoring stays deterministic |
| **Inspect AI** | **Built** as an export format for portability — [method note](interop_inspect.md); not as a methodology |
| **GDPval** | Read the task-construction method; the benchmark itself is out of scope |
| **OpenAI Evals** | Not a fit — capability registry |
| **LangSmith** | Capability worth having, hosting not worth the trade; build the ledger locally |
| **DeepEval / Promptfoo / RAGAS** | Sibling-repo remit |
