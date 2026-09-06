"""Read back a grown handoff fixture and report what is actually in it.

Generation is only half the job. This prints the composition a reviewer needs
before agreeing to spend an API budget running it: how many cases per profile,
how the two baits are distributed, whether the hand-authored and generated
halves are comparable in size and shape, and which records the quality judge
flagged. Everything here is computed from the file, never from the generator's
own claims about what it produced.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generation.handoff import (
    BASE_GATES, FIXTURE_PATH, REQUIRED_FIELDS, collision_hits, tidy_hits,
)


def report(path: Path) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    hand = [r for r in rows if r.get("source", "hand") == "hand"]
    gen = [r for r in rows if r.get("source") == "generated"]

    print(f"{path}  —  {len(rows)} records ({len(hand)} hand, {len(gen)} generated)\n")

    print(f"{'case':9s} {'src':4s} {'profile':17s} {'flds':>4s} {'carry':>5s} "
          f"{'tidy':>4s} {'coll':>4s}  judge")
    for r in rows:
        rec = r["record"]
        print(f'{r["case_id"]:9s} {r.get("source","hand")[:4]:4s} {r["profile"]:17s} '
              f'{len(rec):4d} {len(r["carry_only_fields"]):5d} '
              f'{len(tidy_hits(rec)):4d} {len(collision_hits(rec)):4d}  '
              f'{r.get("judge","")[:46]}')

    print("\nprofile balance")
    for profile, n in Counter(r["profile"] for r in rows).most_common():
        h = sum(1 for r in rows if r["profile"] == profile and r.get("source", "hand") == "hand")
        print(f"  {profile:18s} {n:3d}  ({h} hand, {n - h} generated)")

    print("\nfield-count spread")
    for label, group in (("hand", hand), ("generated", gen)):
        if group:
            counts = sorted(len(r["record"]) for r in group)
            print(f"  {label:10s} min {counts[0]:2d}  median {counts[len(counts)//2]:2d}  "
                  f"max {counts[-1]:2d}")

    # Re-run every gate over the finished file. The generator applied these to
    # candidates; applying them again to what was written is the check that the
    # writing step did not introduce anything the gates never saw.
    failures = [(r["case_id"], g(r)) for r in rows for g in BASE_GATES if g(r)]
    print(f"\ngates re-applied to the written file: "
          f"{'all pass' if not failures else failures}")

    missing = [r["case_id"] for r in rows
               if any(f not in r["record"] for f in REQUIRED_FIELDS)]
    print(f"stage-needed fields present in every record: {'yes' if not missing else missing}")

    flagged = [r["case_id"] for r in rows if str(r.get("judge", "")).startswith(("weak", "reject"))]
    print(f"judge flagged for review: {flagged or 'none'}")

    dupes = [n for n, c in Counter(
        str(r["record"].get("full_name", "")).lower() for r in rows).items() if c > 1]
    print(f"repeated names: {dupes or 'none'}")
    return 1 if failures or missing else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", type=Path, default=FIXTURE_PATH)
    raise SystemExit(report(ap.parse_args().path))


if __name__ == "__main__":
    main()
