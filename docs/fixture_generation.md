# Growing a Fixture Without Losing It

[← Back to README](../README.md) · [Ecosystem note](ecosystem.md) · [Scenario 8](multi_agent_handoff.md)

Every scenario in this library runs on **hand-authored, use-case fixtures** — records and requests written to look like the work a real system does, not items lifted from a public benchmark. That choice buys realism and costs sample size, and the cost is not cosmetic.

At six cases per arm, a two-sided Fisher test cannot return a p-value below **0.002**, and it only gets there when an arm fails *every* case and its comparator fails none. Anything short of total is unreachable:

| Failure rate in the degraded arm (comparator at 0) | p at 6 cases | p at 30 cases |
|---|---|---|
| 20% | 1.000 | **0.024** |
| 33% | 0.455 | **0.001** |
| 50% | 0.182 | **<0.001** |
| 67% | 0.061 | **<0.001** |
| 100% | **0.002** | **<0.001** |

Read it as: at thirty cases an effect that breaks **one record in five** is detectable; at six, only an effect that breaks nearly all of them is, and a 67% effect — a large one, plainly visible in the data — misses the threshold. Several findings in this repo are reported as *underpowered, not null* for exactly that reason, and [scenario 8](multi_agent_handoff.md) says so in its own limitations section.

Estimation improves by roughly the same factor. The same observed rate of 83% carries a 95% interval of [0.44, 0.97] at six cases and [0.66, 0.93] at thirty — the width halves, from 0.53 to 0.26.

---

## The approach: Bloom's shape, not Bloom's metric

[Bloom](https://alignment.anthropic.com/2025/bloom-auto-evals/) (Anthropic, December 2025) turns a plain-English behaviour specification into an evaluation suite through four stages — understand, ideate, roll out, judge. The first two are exactly what is needed here. The last one is not.

**Bloom's judge scores the rollout. This repo's never does.** Bloom's headline metric is an elicitation rate produced by an LLM judge reading transcripts, which is the right design for open-ended behaviour. It is the wrong design here, because it would put a model on both ends of the pipeline — one inventing the test, another marking it — and a judge that shares an author's blind spot produces agreement, not evidence.

So the loop implemented in [`generation/bloom.py`](../generation/bloom.py) is:

1. **Behaviour spec** — prose a reviewer can read and disagree with, not a schema.
2. **Ideate** — the seed cases become few-shot templates; the model proposes more.
3. **Gate** — every proposal passes mechanical checks before it is allowed into a fixture.
4. **Judge** — a *second* model rates the proposed fixtures for realism and on-profile-ness. Its verdicts are stored as provenance and used to decide what a human looks at first. They gate nothing.

Scoring the actual runs stays where it was: deterministic comparison against ground truth.

---

## The gates are the contract; the prompt is a suggestion

A generator asked for "more cases like these" reliably produces more cases that *read* like these and quietly stop *being* like these. The control profile picks up a little of the bait. Field counts converge on the mean. Three records share a surname. None of that is visible by reading the output. All of it is visible to a gate.

For [scenario 8](multi_agent_handoff.md) the gates are in [`generation/handoff.py`](../generation/handoff.py):

| Gate | Rejects |
|---|---|
| Shape | Missing record, missing carry-only list, a rationale too short to review |
| Values | Non-scalar or empty values, field names that are not plain snake_case, field counts outside 12–24 |
| Stage needs | A record missing a field some stage needs for its own job — this one is load-bearing, see below |
| Carry-only | Fewer than two carry-only fields, a carry-only field not in the record, or a carry-only field a stage actually needs |
| Safety | A name colliding with the scenario's sanctions watchlist, an email off a `.invalid` domain, a URL pointing somewhere real |
| Uniqueness | A repeated name or tax identifier, across profiles as well as within one |
| **Profile** | A record that does not instantiate the mechanism its profile claims |

The **stage-needs** gate exists because of how the scorer divides. `own_fields_retained` is measured against what each stage needs regardless of whether the record supplied it, so a generated record missing `tax_id` would cap the screening stage at 4/5 forever and drag the metric down for a reason that has nothing to do with the handoff. That is the class of bug generation introduces: not a bad record, a quietly wrong denominator.

The **profile** gate is the one the whole exercise depends on, and it is mechanical rather than semantic:

- `clean_values` — the control. Zero tidying bait, at most one delimiter collision (times contain colons; that is unavoidable and harmless).
- `correction_bait` — at least three *declared* bait fields, each with a `tempting_edit` saying what a tidying model would emit instead, and at least two of them detectable by pattern.
- `collision_bait` — at least four values containing a token the pipeline uses as a delimiter, and **zero** tidying bait, so the two variables stay separated.
- `carry_only_heavy` — at least 35% of fields are ones no stage needs, and no bait of either kind.

Requiring the generator to *declare* its bait, and to say what the tidied version would look like, is what makes `correction_bait` checkable at all. A pattern can catch a non-ISO date and a lower-case nationality; it cannot catch `Self-emploied - marine surveior`. The declaration can, and it is verified: the stated edit must differ from the original, and must cover exactly the declared fields.

---

## What the gates actually caught

Forty proposals, thirty-one accepted (twenty-four kept, seven surplus), nine rejected:

| Reason | Count |
|---|---|
| Carry-only share below the profile's threshold | 4 |
| `carry_only_fields` names a field not in the record | 2 |
| Carry-only record also carries bait — two variables at once | 2 |
| Collision record also carries tidying bait | 1 |

Every rejection landed on a profile boundary, and half of them on one profile: **`carry_only_heavy` rejected 8 of its 16 proposals.** That is the convergence failure, caught in the act — asked for records that are mostly dead weight, the generator kept drifting back toward a conventional field mix.

Two profiles rejected nothing. That is a caveat, not a clean bill of health: a gate that never fires has not been shown to work.

## What the judge got wrong, and why it is recorded rather than acted on

The first quality-gate pass rejected **17 of 24** records, on two criteria it was never given:

- It validated national-identifier checksums — CUIT, PESEL, NIF, SIN — and rejected records whose identifiers would not pass a real registry.
- It read `carry_only_heavy` as meaning a *majority* of fields, and rejected all six for having 35–40%.

Both readings are reasonable in the abstract and wrong here. A national identifier that validates against a real registry is a **defect** in a fixture published to a public repository, not a merit; and the profile threshold is 35% by design. The judge was re-run with criteria that say both things explicitly, after which it returned 18 ok, 1 weak, 5 reject.

The five remaining rejections are all of the form *"this identifier shape is not plausible for that jurisdiction"* — a cosmetic realism complaint that cannot affect what the scenario measures, since the only question asked of these values is whether they arrive byte-identical. All twenty-four records were kept. Each flagged record carries the judge's verdict in the fixture, so a reader can see what was flagged and that a person overrode it.

The general lesson is the one that motivated keeping the judge off the scoring path in the first place: **an LLM quality gate applies the criteria it infers, not the criteria you wrote.** Here that was visible because the design has explicit numeric thresholds to check it against. On a scoring path, it would not have been visible at all.

---

## Provenance, and the check it buys

Every record carries `source`: `hand` for the six originals, `generated` for the rest. This is not bookkeeping. It is the only available check on whether generation shifted the difficulty of the set: **every result can be split hand-vs-generated, and if the two halves disagree, the generated half is the suspect.**

The six-record set the published results were produced against is frozen at [`scenarios/fixtures/multi_agent_handoff.v1.jsonl`](../scenarios/fixtures/multi_agent_handoff.v1.jsonl), so those runs stay reproducible against the fixture that produced them.

## The risk this does not solve

Generation at scale reintroduces author bias — cases produced by the same process that scores them, at thirty times the volume. Gates constrain the *shape* of a case; they cannot tell you the four profiles are the right four.

Scenario 8 was chosen for the pilot partly because it minimises this: its ground truth **is** the record, so a generator cannot make the test easier without also moving the answer key to match. There is no target to move. That property does not generalise — in a scenario where the correct answer is authored separately from the input, generation can drift the two apart silently — which is why the next fixture to grow needs the external reproduction track that [Tool / MCP Abuse](tool_mcp_abuse.md) already runs, generalised.

---

## Running it

```bash
python -m generation.handoff --n-per-profile 6 --dry-run    # propose and report, write nothing
python -m generation.handoff --n-per-profile 6              # write the fixture
python -m generation.handoff --rejudge scenarios/fixtures/multi_agent_handoff.jsonl
python -m generation.inspect_handoff                        # composition, gates re-applied, judge flags
```

`--dry-run` is the one to reach for first: it pays for generation, prints the rejection table and the profile balance, and writes nothing. Sample size is a parameter everywhere; nothing about `n` is baked in.

[`generation/inspect_handoff.py`](../generation/inspect_handoff.py) re-applies every gate to the **written file** rather than to the candidates, which is the check that the writing step did not introduce something the gates never saw.
