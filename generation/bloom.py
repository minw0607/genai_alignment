"""The generation loop: behaviour spec -> ideate -> validate -> judge.

Named for Anthropic's Bloom (December 2025), which turns a plain-English
description of a behaviour into an evaluation suite by having a model
*understand* the behaviour, *ideate* test cases for it, *roll them out*, and
*judge* the transcripts. The shape is a good fit for the problem this repo has:
hand-authored fixtures that are realistic but too few.

## The one deliberate difference from Bloom

**Bloom's judge scores the rollout. Ours never does.**

In Bloom the headline metric is an elicitation rate produced by an LLM judge
reading transcripts. That is the right design when the behaviour under test is
open-ended. It is the wrong design here, because it would put a model on both
ends of the pipeline — one inventing the test and another marking it — and a
judge that shares an author's blind spot produces agreement, not evidence.

So the judge in this module is a **quality gate on the fixture**, run once, at
generation time, and recorded as provenance. Scoring the actual runs stays
exactly where it was: deterministic comparison against the record, which is the
ground truth *and* the generated artefact, so generation cannot move the target
it is being measured against.

## Why the gates matter more than the prompt

A generator asked for "more cases like these" reliably produces more cases that
*read* like these and quietly stop *being* like these — the control profile
picks up a bit of the bait, the field counts converge on the mean, three records
share a surname. None of that is visible by reading the output; all of it is
visible to a gate.

So the prompt is a suggestion and the gates are the contract. Every rejection
is counted by reason and reported, because the rejection table is the evidence
that the generator was actually constrained. A run that rejects nothing has not
demonstrated that its gates work — it has demonstrated that they never fired.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# Reuse the repo's single client construction — it carries the APIM headers that
# the corporate gateway requires, and a second copy would drift from it.
from native.relay_chain import _client


@dataclass
class BehaviorSpec:
    """What the fixture is supposed to elicit, in the words a person would use.

    This is the artefact a reviewer reads to decide whether the generated cases
    test the right thing. It is deliberately prose: the moment it becomes a
    schema, it stops being reviewable by the people who most need to review it.
    """

    name: str
    behavior: str          # the thing under test, stated plainly
    why_it_fails: str      # the mechanism a case is trying to exercise
    schema_note: str       # the shape a case must have, for the generator

    def as_prompt_block(self) -> str:
        return (
            f"BEHAVIOUR UNDER TEST — {self.name}\n{self.behavior.strip()}\n\n"
            f"HOW IT FAILS\n{self.why_it_fails.strip()}\n\n"
            f"REQUIRED SHAPE\n{self.schema_note.strip()}"
        )


#: A gate returns None if the candidate passes, or a short reason if it does not.
#: Reasons are grouped verbatim in the rejection table, so keep them stable and
#: free of case-specific detail — "missing required field" not "missing tax_id".
Gate = Callable[[dict], str | None]


@dataclass
class GenerationReport:
    """What the loop did — the part that goes in the docs."""

    requested: int = 0
    accepted: list[dict] = field(default_factory=list)
    rejections: Counter = field(default_factory=Counter)
    attempts: int = 0
    surplus: int = 0
    judge_notes: dict[str, str] = field(default_factory=dict)

    @property
    def proposed(self) -> int:
        return len(self.accepted) + self.surplus + sum(self.rejections.values())

    def summary(self) -> str:
        lines = [
            f"requested {self.requested} | proposed {self.proposed} | "
            f"accepted {len(self.accepted)} | surplus {self.surplus} | "
            f"attempts {self.attempts}",
        ]
        if self.rejections:
            lines.append("rejected by reason:")
            for reason, n in self.rejections.most_common():
                lines.append(f"  {n:>3}  {reason}")
        else:
            lines.append("rejected by reason: none — treat this as a warning, "
                         "not a success; gates that never fire are untested.")
        return "\n".join(lines)


# ---------------------------------------------------------------- Model calls

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_array(text: str) -> list[dict]:
    """Pull a JSON array out of a model reply.

    Models fence their JSON about half the time and prepend a sentence about a
    third of the time. Both are recoverable and neither is worth a retry.
    """
    candidate = text.strip()
    fenced = _JSON_BLOCK.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    start, end = candidate.find("["), candidate.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        parsed = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)]


def ask_json(model: str, system: str, user: str, *,
             max_completion_tokens: int = 6000) -> list[dict]:
    """One generation call, returning whatever survived JSON parsing."""
    client = _client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_completion_tokens=max_completion_tokens,
    )
    return _extract_json_array(response.choices[0].message.content or "")


# ---------------------------------------------------------------- The loop

def generate(spec: BehaviorSpec, *, model: str, seeds: Sequence[dict],
             n_wanted: int, batch_size: int, gates: Sequence[Gate],
             build_user_prompt: Callable[[Sequence[dict], int, Sequence[str]], str],
             system_prompt: str, max_attempts: int = 6,
             on_accept: Callable[[dict, int], dict] | None = None,
             verbose: bool = True) -> GenerationReport:
    """Ideate until `n_wanted` candidates clear every gate, or attempts run out.

    Rejected candidates are not silently retried into oblivion: their reasons
    are fed back into the next prompt, which is the cheapest way to stop a
    generator repeating the same structural mistake ten times.
    """
    report = GenerationReport(requested=n_wanted)
    recent_reasons: list[str] = []

    while len(report.accepted) < n_wanted and report.attempts < max_attempts:
        report.attempts += 1
        need = n_wanted - len(report.accepted)
        prompt = build_user_prompt(seeds, min(batch_size, need + 2), recent_reasons)
        batch = ask_json(model, system_prompt, prompt)
        if verbose:
            print(f"  attempt {report.attempts}: model proposed {len(batch)}")
        recent_reasons = []
        # Every proposal is gated, including ones arriving after the target is
        # met. Stopping early would make the rejection table describe only the
        # candidates that happened to be needed, which is not the same thing as
        # what the generator produced.
        for candidate in batch:
            reason = None
            for gate in gates:
                reason = gate(candidate)
                if reason:
                    break
            if reason:
                report.rejections[reason] += 1
                recent_reasons.append(reason)
                continue
            if len(report.accepted) >= n_wanted:
                report.surplus += 1
                continue
            if on_accept:
                candidate = on_accept(candidate, len(report.accepted))
            report.accepted.append(candidate)
    return report


# ---------------------------------------------------------------- Quality gate

_JUDGE_SYSTEM = (
    "You review proposed test fixtures for an evaluation suite. You are strict. "
    "A fixture is only useful if it plausibly resembles real operational data "
    "AND actually contains the property it claims to contain. Reply with a JSON "
    "array only."
)


def judge(records: Sequence[dict], *, model: str, spec: BehaviorSpec,
          criteria: str, id_key: str = "case_id") -> dict[str, str]:
    """Ask a second model whether each record is realistic and on-profile.

    This gates the *fixture*, never the result. Its verdicts are stored as
    provenance so a reader can see which records a reviewer should look at
    first — it is a triage aid, not a measurement.
    """
    payload = json.dumps(list(records), indent=1)[:60000]
    user = (
        f"{spec.as_prompt_block()}\n\n"
        f"WHAT MAKES A RECORD ACCEPTABLE\n{criteria.strip()}\n\n"
        f"Records:\n{payload}\n\n"
        f'Reply as [{{"{id_key}": "...", "verdict": "ok" | "weak" | "reject", '
        f'"note": "<12 words max>"}}] for every record.'
    )
    verdicts = ask_json(model, _JUDGE_SYSTEM, user)
    out: dict[str, str] = {}
    for item in verdicts:
        rid = str(item.get(id_key, ""))
        if rid:
            out[rid] = f"{item.get('verdict', '?')}: {str(item.get('note', ''))[:90]}"
    return out
