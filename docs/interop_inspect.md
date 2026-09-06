# Exporting to Inspect AI

[← Back to README](../README.md) · [Ecosystem note](ecosystem.md) · [Scenario 6](boundary_permission.md)

The standing weakness of use-case-grounded evaluation is that **nobody outside the repository can run it.** A public benchmark is portable by construction — its cases *are* the artefact. A hand-authored scenario is not: the cases live in a fixture, the system under test lives in a local harness, and the scoring lives in a module that knows both. Reproduction elsewhere is possible in principle and inconvenient enough in practice that it does not happen.

[Inspect AI](https://inspect.aisi.org.uk/) (UK AI Security Institute) is the one widely-used framework in the landscape built for safety and alignment evaluation rather than capability scoring, and it is treated here as an **export target, not a source of method** — see the [ecosystem note](ecosystem.md) for why its runtime does not replace the contrast ladder, the floor arm, or the power floor.

Piloted on [Boundary / Permission](boundary_permission.md), chosen because it is native to this repo with no sibling-repo dependency and has the cleanest deterministic scorer.

---

## What gets written

```bash
python -m interop.inspect_export boundary_permission
python -m interop.inspect_export boundary_permission --validate   # needs inspect-ai installed
```

Into [`interop/exports/inspect/`](../interop/exports/inspect/) — committed rather than under the gitignored `outputs/`, because the whole point is that somebody else can pick it up from the repository:

| File | What it is |
|---|---|
| `boundary_permission_dataset.jsonl` | All 40 cases as Inspect `Sample` objects — `id`, `input`, `target`, `metadata`. Loads with `json_dataset()`. |
| `boundary_permission_task.py` | A task stub wiring that dataset to a deterministic scorer, with the solver left as a placeholder. |
| `boundary_permission_<policy>_<menu>.eval.json` | One eval log per arm — 120 runs each — readable by `inspect view`. |

**The dataset is the half that matters.** It carries the actual intellectual content: an Inspect user can run **their own** agent against **these** cases, which is the reproduction path that does not currently exist. The eval logs are the smaller win — they let our numbers be opened next to a reader's own run instead of only read in our report.

## One log per arm, not one per scenario

Inspect's log holds a single run of a single task. This repo's arms are separate conditions whose entire purpose is to be compared:

| Log | Runs | Accuracy |
|---|---|---|
| `enforced_full` | 120 | 1.000 |
| `enforced_minimal` | 120 | 1.000 |
| `unguarded_full` | 120 | 0.467 |

Flattening those into one log would produce a headline of 0.82 — a number averaged across conditions that were never meant to be averaged, and precisely the conflation the contrast ladder exists to prevent. The arm is recorded in each log's `eval.metadata` and in every sample's metadata, so the comparison survives the export.

## What the export deliberately does not do

**It does not emit a runnable port.** Doing so would mean reimplementing the tool harness, the tool backend and the authorization state inside Inspect's solver abstraction — a second implementation of the system under test, which would then need its own equivalence testing against the first. The honest boundary is: *we export the cases and the scores; the reader brings the agent.* The task stub shows the wiring and says so in its own docstring.

**It does not flatten the outcome space silently.** Inspect's scale is binary; this scenario's is not. `over_refusal` — declining something the agent was authorized to do — is a real failure but not a boundary violation, and collapsing the two would hide the trade the scenario exists to measure. Over-refusals score as incorrect *and say which kind of failure they were* in an explanation that survives into `inspect view`.

## The schema was verified, not assumed

Every emitted log was loaded back through `inspect_ai.log.read_eval_log`, the dataset through `json_dataset()`, and the task stub was executed. That found two errors reading the documentation had not:

- `eval.scorers[].metrics` is a **list** of metric definitions, not a mapping.
- `input` is **required** on every sample, and had to be carried across from the fixture — the results file stores outcomes, not the request text.

Both are fixed, and the round-trip is what `--validate` reproduces.

`inspect_ai` is deliberately **not a dependency**: an export path should not require installing the thing being exported to, and nobody should need Inspect to run a scenario here. The exporter writes plain dicts, and `--validate` degrades to a clear message when Inspect is absent. The release the output was last verified against is recorded as `VALIDATED_AGAINST` in [`interop/inspect_export.py`](../interop/inspect_export.py); if a future version rejects these logs, that flag is what will say so.

## Adding another scenario

One dict in [`interop/inspect_export.py`](../interop/inspect_export.py): fixture path, results path, which column is the id / the input / the target, which columns become metadata, which columns identify an arm, and a `score_fn` mapping that scenario's outcome vocabulary onto Inspect's `C` / `I` / `N`. The scorer source string is emitted into the task stub, so the exported scorer stays readable rather than becoming a black box.

The mapping function is the part worth thinking about rather than copying. Every scenario in this library reports something richer than correct/incorrect, and how that collapses is a judgement — one that should be written down in the `score_fn` docstring where a reader will find it, not buried in the export.

## Confidentiality

Model identifiers never leave as deployment names: every export writes the generic labels from `reporting/display.py`, and free text is scrubbed of provider strings on the way out. Model replies quote raw API error text often enough that this is not a theoretical concern — it has leaked into a committed report in this repo before.
