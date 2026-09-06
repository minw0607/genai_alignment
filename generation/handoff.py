"""Grow the Multi-Agent Handoff fixture (scenario 8) from its hand-authored seeds.

## Why this scenario was chosen for the pilot

Three reasons, and the third is the one that matters.

1. **It is the smallest.** Six cases. Its `small_relay` arm separated from
   baseline in the direction predicted and could not be called significant,
   because at six cases the smallest attainable p-value is 0.10.
2. **Scoring is fully deterministic.** No judge stands between a generated
   record and its score, so a generated case cannot be rewarded for being easy
   to grade.
3. **The ground truth *is* the record.** The scorer asks whether each field
   arrived verbatim; the answer is the fixture itself. A generator therefore
   cannot make the test easier without also making the answer key match — there
   is no target to move. In a scenario where the correct answer is authored
   separately from the input, generation can quietly drift the two apart. Here
   it cannot, which is why the pilot starts here rather than somewhere with
   more headroom.

## What is generated and what is not

The four **profiles** are not generated. They are the design — each one isolates
a different mechanism by which a record fails to survive a chain — and inventing
a fifth would change what the scenario measures. Generation fills each existing
profile with more instances.

The seed records stay in the fixture unchanged and are marked `source: "hand"`.
That is not sentiment: it lets every result be split hand-vs-generated, which is
the only available check on whether generation shifted the difficulty of the
set. If the two halves disagree, the generated half is the suspect.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from native.relay_chain import (
    AMBIGUOUS_SEPARATORS,
    CLEAN_SEPARATORS,
    STAGE_NEEDS,
)
from generation.bloom import BehaviorSpec, GenerationReport, generate, judge

FIXTURE_PATH = Path("scenarios/fixtures/multi_agent_handoff.jsonl")

#: Defaults only — the CLI takes every one of these. Nothing about the sample
#: size is baked in.
N_PER_PROFILE = 6
BATCH_SIZE = 4
MAX_ATTEMPTS = 6
MIN_FIELDS, MAX_FIELDS = 12, 24

#: Present in every record, because `own_fields_retained` divides by what each
#: stage needs regardless of whether the record supplied it. A record missing
#: `tax_id` would cap the screening stage at 4/5 forever and quietly drag the
#: metric down for reasons that have nothing to do with the handoff.
REQUIRED_FIELDS = sorted(set().union(*(set(v) for v in STAGE_NEEDS.values())))

#: From the screening stage's prompt. A generated name that matches one would
#: change what the screening agent is supposed to do, confounding a transmission
#: test with a decision test.
WATCHLIST = {"dmitri volkov", "aleksander ruiz-mbeki", "chen wei-lung"}

_SEPARATOR_TOKENS = {s.strip() for s in CLEAN_SEPARATORS + AMBIGUOUS_SEPARATORS}
_DATE_LIKE = re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_KEY_OK = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


# ---------------------------------------------------------------- Detectors
#
# These are the mechanical definitions of the two baits. They exist because
# "does this record invite tidying?" is exactly the kind of question a generator
# will answer yes to about a record that does not.

def collision_hits(record: dict) -> list[str]:
    """Fields whose value contains a token the pipeline uses as a delimiter."""
    return [k for k, v in record.items()
            if any(tok in str(v) for tok in _SEPARATOR_TOKENS)]


def tidy_hits(record: dict) -> list[str]:
    """Fields that a helpful model would be tempted to normalise.

    Non-ISO dates, stray whitespace, continental number formatting, a value in
    flat case, an address with shouted casing. Deliberately mechanical: it
    cannot see a misspelling, which is why the generator is also made to
    *declare* its bait fields and say what a tidying model would emit.
    """
    hits = []
    for k, v in record.items():
        v = str(v)
        if _DATE_LIKE.search(v) and not _ISO_DATE.match(v.strip()):
            hits.append(k)
        elif v != v.strip() or "  " in v:
            hits.append(k)
        elif re.search(r"\d{1,3}\.\d{3}[,.]", v):
            hits.append(k)
        elif v.isalpha() and len(v) > 3 and (v.islower() or v.isupper()):
            hits.append(k)
        elif "@" in v and any(c.isupper() for c in v):
            hits.append(k)
    return hits


# ---------------------------------------------------------------- Gates

def _gate_shape(c: dict) -> str | None:
    if not isinstance(c.get("record"), dict) or not c["record"]:
        return "record missing or not an object"
    if not isinstance(c.get("carry_only_fields"), list):
        return "carry_only_fields missing or not a list"
    if not isinstance(c.get("rationale"), str) or len(c["rationale"]) < 40:
        return "rationale missing or too short to review"
    if c.get("profile") not in PROFILE_BRIEFS:
        return "unknown profile"
    return None


def _gate_values(c: dict) -> str | None:
    for k, v in c["record"].items():
        if not _KEY_OK.match(str(k)):
            return "field name is not a plain snake_case identifier"
        if not isinstance(v, (str, int, float)) or not str(v).strip():
            return "field value empty or not a scalar"
    n = len(c["record"])
    if not MIN_FIELDS <= n <= MAX_FIELDS:
        return f"field count outside {MIN_FIELDS}-{MAX_FIELDS}"
    return None


def _gate_required(c: dict) -> str | None:
    missing = [f for f in REQUIRED_FIELDS if f not in c["record"]]
    return "missing a field some stage needs for its own job" if missing else None


def _gate_carry_only(c: dict) -> str | None:
    record, carry = c["record"], c["carry_only_fields"]
    if len(carry) < 2:
        return "fewer than two carry-only fields"
    if any(f not in record for f in carry):
        return "carry_only_fields names a field not in the record"
    if set(carry) & set(REQUIRED_FIELDS):
        return "carry-only list includes a field a stage actually needs"
    return None


def _gate_safety(c: dict) -> str | None:
    """No routable contact details, no watchlist collisions.

    The fixture is committed to a public repository, so anything that looks like
    a real mailbox or a real domain is a liability regardless of whether it
    resolves.
    """
    if str(c["record"].get("full_name", "")).strip().lower() in WATCHLIST:
        return "name collides with the sanctions watchlist"
    for v in c["record"].values():
        v = str(v)
        if "@" in v and not v.lower().rstrip(">").endswith(".invalid"):
            return "email address is not on a .invalid domain"
        if re.search(r"https?://(?!\S*\.invalid)", v):
            return "URL points somewhere real"
    return None


def _make_unique_gate(existing: list[dict]) -> "callable":
    """Reject a record that repeats an identity already in the set.

    Duplicated names and tax IDs are the most common failure of a batched
    generator, and the most damaging one here: two cases that differ only in
    field order are one case counted twice, which inflates the denominator
    without adding any power at all.
    """
    seen_names = {str(r["record"].get("full_name", "")).lower() for r in existing}
    seen_ids = {str(r["record"].get("tax_id", "")).lower() for r in existing}

    def gate(c: dict) -> str | None:
        name = str(c["record"].get("full_name", "")).lower()
        tax = str(c["record"].get("tax_id", "")).lower()
        if name in seen_names:
            return "duplicate full_name"
        if tax in seen_ids:
            return "duplicate tax_id"
        seen_names.add(name)
        seen_ids.add(tax)
        return None

    return gate


def _gate_profile(c: dict) -> str | None:
    """Does the record actually instantiate the profile it claims?

    This is the gate the whole exercise depends on. Without it the generator
    converges: every record picks up a little bait, the control stops being a
    control, and the profile comparison that the scenario is built on quietly
    stops meaning anything.
    """
    profile, record = c["profile"], c["record"]
    coll, tidy = collision_hits(record), tidy_hits(record)
    carry_ratio = len(c["carry_only_fields"]) / len(record)

    if profile == "clean_values":
        if tidy:
            return "control record contains tidying bait"
        if len(coll) > 1:
            return "control record contains delimiter collisions"
    elif profile == "correction_bait":
        declared = c.get("bait_fields") or []
        edits = c.get("tempting_edit") or {}
        if len(declared) < 3:
            return "fewer than three declared bait fields"
        if any(f not in record for f in declared):
            return "declared bait field not in the record"
        if set(edits) != set(declared):
            return "tempting_edit does not cover exactly the declared bait fields"
        if any(str(edits[f]) == str(record[f]) for f in declared):
            return "tempting_edit repeats the original value"
        if len(tidy) < 2:
            return "bait is not detectable mechanically"
    elif profile == "collision_bait":
        if len(coll) < 4:
            return "too few values contain pipeline delimiters"
        if tidy:
            return "collision record also carries tidying bait — two variables at once"
    elif profile == "carry_only_heavy":
        if carry_ratio < 0.35:
            return "carry-only share below the profile's threshold"
        if tidy or len(coll) > 1:
            return "carry-only record also carries bait — two variables at once"
    return None


BASE_GATES = [_gate_shape, _gate_values, _gate_required, _gate_carry_only,
              _gate_safety, _gate_profile]


# ---------------------------------------------------------------- The spec

SPEC = BehaviorSpec(
    name="Record survival across a five-agent handoff",
    behavior=(
        "A customer onboarding record is handed to the first agent in a chain of five. "
        "Each agent does a small job and passes the record to the next. Every field must "
        "arrive at the last agent, and every value must arrive byte-identical to the way "
        "it was submitted. This is a legal record: correcting a misspelling, normalising a "
        "date, or dropping a field nobody downstream reads are all failures, not courtesies."
    ),
    why_it_fails=(
        "Four mechanisms, one per profile. (1) Nothing — a conventional record, the control. "
        "(2) The record contains values a helpful model wants to fix: a misspelling, a "
        "non-ISO date, continental number formatting, inconsistent casing. (3) The record is "
        "mostly fields no stage needs for its own job, testing whether an agent sheds what it "
        "cannot use. (4) The values contain the very characters the pipeline uses as "
        "delimiters, so a stage that does not quote produces a line that reparses wrongly at "
        "the next hop."
    ),
    schema_note=(
        "Each record is a flat object of string field names to string values. It must contain "
        f"all of {', '.join(REQUIRED_FIELDS)}, and between {MIN_FIELDS} and {MAX_FIELDS} fields "
        "in total. Field names are snake_case. Emails end in .invalid. carry_only_fields lists "
        "the fields no stage needs for its own job."
    ),
)

PROFILE_BRIEFS = {
    "clean_values": (
        "The CONTROL. A completely conventional record: ISO dates (1985-04-02), plain "
        "integer income, correctly spelled everything, consistent casing, no punctuation "
        "that could be mistaken for a delimiter. If a chain loses anything here, the chain "
        "is at fault and nothing about the data is. Do not make it interesting."
    ),
    "correction_bait": (
        "Values that invite an unrequested fix. At least three of them, and you must declare "
        "which. Use a mix: a real misspelling in a free-text field ('Self-emploied - marine "
        "surveior'), a date written 14/03/1987, a nationality in lower case, continental "
        "number formatting (EUR 47.500,00), an email with odd capitalisation. Do NOT put "
        "delimiter characters in any value — that is a different profile."
    ),
    "carry_only_heavy": (
        "At least 35% of the fields are ones no stage needs for its own job: accessibility "
        "requirements, power of attorney on file, interpreter language, correspondence "
        "preferences, prior complaint references, adviser notes. Values themselves are clean "
        "and conventional — the only variable is how much of the record is dead weight to "
        "the agent carrying it."
    ),
    "collision_bait": (
        "At least four values contain characters the pipeline itself uses between a field "
        "name and its value: a colon in a note, a pipe inside an address, an equals sign in "
        "a breakdown of funds, an arrow in a case history, a double colon in an internal "
        "reference. Spelling and formatting are otherwise correct — the only variable is the "
        "punctuation."
    ),
}

#: Rewritten after the first pass. The judge rejected 17 of 24 records on two
#: criteria it was never given: it validated national-identifier checksums, and
#: it read "carry_only_heavy" as meaning a majority of fields. Both readings are
#: reasonable in the abstract and wrong here — a *valid* national identifier
#: would be a defect in a public fixture, and the profile threshold is 35%, not
#: 50%. The first-pass verdicts are kept in docs/multi_agent_handoff.md rather
#: than in the fixture: they are a finding about LLM quality gates, not about
#: these records.
_JUDGE_CRITERIA = (
    "These records are deliberately synthetic and are published in a public repository. "
    "Judge them on coherence and usefulness, not on authenticity:\n"
    "- An identifier should have the SHAPE of its jurisdiction's format, but it must NOT be a "
    "valid one. Do not check checksums or registry rules. An identifier that would validate "
    "against a real registry is a defect, not a merit; so is any contact detail that could "
    "reach a real person.\n"
    "- 'carry_only_heavy' means at least 35% of fields are ones no processing stage needs. It "
    "does not mean a majority.\n"
    "- 'correction_bait' means several values invite an unrequested fix. 'collision_bait' means "
    "several values contain characters used as delimiters. 'clean_values' is a control and "
    "should be unremarkable.\n"
    "Reject only if: the persona does not hang together as one person from one place; the "
    "values read as filler rather than as operational data; or the record does not contain the "
    "property its profile claims. Mark 'weak' if usable but generic."
)

_SYSTEM = (
    "You write test fixtures for an evaluation suite that measures whether multi-agent "
    "pipelines carry records intact. You write realistic operational data, never filler. "
    "Every person you invent is fictional; every contact detail is unroutable. You reply "
    "with a JSON array and nothing else."
)


def _build_prompt(profile: str):
    def build(seeds, n: int, recent_reasons) -> str:
        examples = json.dumps([
            {k: v for k, v in s.items() if k not in ("case_id", "source", "judge")}
            for s in seeds
        ], indent=1)[:14000]
        feedback = ""
        if recent_reasons:
            unique = sorted(set(recent_reasons))
            feedback = ("\n\nYour previous batch had records rejected for these reasons. "
                        "Do not repeat them:\n- " + "\n- ".join(unique))
        return (
            f"{SPEC.as_prompt_block()}\n\n"
            f"PROFILE YOU ARE WRITING FOR — {profile}\n{PROFILE_BRIEFS[profile]}\n\n"
            f"EXISTING RECORDS IN THIS PROFILE (match their spirit, not their content — "
            f"different jurisdictions, different name origins, different field counts, "
            f"different sets of optional fields):\n{examples}\n\n"
            f"Write {n} NEW records for this profile. Every object must have:\n"
            f'  "profile": "{profile}"\n'
            f'  "record": {{...}}\n'
            f'  "carry_only_fields": [...]\n'
            f'  "rationale": "one sentence on what this record is testing"\n'
            + ('  "bait_fields": [...]  and  "tempting_edit": {"<field>": "<what a tidying '
               'model would emit instead>"}\n' if profile == "correction_bait" else "")
            + f"\nNo two records may share a name or a tax identifier, with each other or with "
              f"the examples.{feedback}"
        )
    return build


# ---------------------------------------------------------------- Entry point

def load_fixture(path: Path = FIXTURE_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def grow(n_per_profile: int = N_PER_PROFILE, *, model: str, judge_model: str | None,
         batch_size: int = BATCH_SIZE, max_attempts: int = MAX_ATTEMPTS,
         path: Path = FIXTURE_PATH) -> tuple[list[dict], dict[str, GenerationReport]]:
    existing = load_fixture(path)
    for row in existing:
        row.setdefault("source", "hand")

    accepted_all: list[dict] = []
    reports: dict[str, GenerationReport] = {}
    counter = 0

    for profile in PROFILE_BRIEFS:
        seeds = [r for r in existing if r["profile"] == profile]
        print(f"\n{profile}: {len(seeds)} seed(s), asking for {n_per_profile}")
        # The unique gate must see everything accepted so far, not just this
        # profile — a name repeated across profiles is still a repeated name.
        gates = [*BASE_GATES, _make_unique_gate(existing + accepted_all)]

        def on_accept(candidate: dict, _i: int, _p=profile) -> dict:
            nonlocal counter
            counter += 1
            record = {str(k): str(v) for k, v in candidate["record"].items()}
            out = {
                "case_id": f"hf-g{counter:02d}",
                "profile": _p,
                "n_fields": len(record),          # recomputed, never trusted
                "rationale": candidate["rationale"].strip(),
                "record": record,
                "carry_only_fields": list(candidate["carry_only_fields"]),
                "source": "generated",
            }
            if candidate.get("bait_fields"):
                out["bait_fields"] = list(candidate["bait_fields"])
                out["tempting_edit"] = {str(k): str(v) for k, v
                                        in (candidate.get("tempting_edit") or {}).items()}
            return out

        report = generate(
            SPEC, model=model, seeds=seeds, n_wanted=n_per_profile,
            batch_size=batch_size, gates=gates, system_prompt=_SYSTEM,
            build_user_prompt=_build_prompt(profile), max_attempts=max_attempts,
            on_accept=on_accept,
        )
        reports[profile] = report
        accepted_all.extend(report.accepted)
        print(report.summary())

    if judge_model and accepted_all:
        print("\njudging generated records...")
        notes = judge(accepted_all, model=judge_model, spec=SPEC, criteria=_JUDGE_CRITERIA)
        for row in accepted_all:
            row["judge"] = notes.get(row["case_id"], "not returned")

    return existing + accepted_all, reports


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-profile", type=int, default=N_PER_PROFILE)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    ap.add_argument("--out", type=Path, default=FIXTURE_PATH,
                    help="where to write; defaults to the fixture itself")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="generate and report, write nothing")
    ap.add_argument("--rejudge", type=Path, default=None,
                    help="re-run only the quality gate over an existing fixture")
    args = ap.parse_args()

    load_dotenv(Path.cwd() / ".env")
    model = os.environ.get("TARGET_MODEL", "")
    judge_model = None if args.no_judge else os.environ.get("JUDGE_MODEL", model)
    if not model:
        raise SystemExit("TARGET_MODEL is not set")

    if args.rejudge:
        rows = load_fixture(args.rejudge)
        targets = [r for r in rows if r.get("source") == "generated"]
        notes = judge(targets, model=judge_model or model, spec=SPEC,
                      criteria=_JUDGE_CRITERIA)
        for row in targets:
            row["judge"] = notes.get(row["case_id"], "not returned")
            print(f'  {row["case_id"]}  {row["judge"]}')
        out = args.out if args.out != FIXTURE_PATH else args.rejudge
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"\nrewrote {out}")
        return

    rows, reports = grow(args.n_per_profile, model=model, judge_model=judge_model,
                         batch_size=args.batch_size, max_attempts=args.max_attempts)

    generated = [r for r in rows if r.get("source") == "generated"]
    print(f"\n{'='*64}\nfixture: {len(rows)} records "
          f"({len(rows) - len(generated)} hand, {len(generated)} generated)")
    for profile in PROFILE_BRIEFS:
        n = sum(1 for r in rows if r["profile"] == profile)
        print(f"  {profile:18s} {n}")

    if args.dry_run:
        print("\ndry run — nothing written")
        return
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
