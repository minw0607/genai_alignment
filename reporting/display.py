"""Generic, non-identifying labels for reports and notebook output.

The real `TARGET_MODEL` / `JUDGE_MODEL` deployment names and `OPENAI_API_VERSION`
read from `.env` are the user's own confidential test configuration, not
something every reader of a report or a committed sample HTML file should see
(each user of this repo has their own). They're used as-is for actual API
calls, but must never appear in a notebook's displayed output, a saved report,
or a committed `docs/samples/*.html` file — use these generic stand-ins there
instead.
"""

GENERIC_MODEL_NAME = "gpt-5.5"
GENERIC_JUDGE_MODEL_NAME = "gpt-5.4"  # distinct from GENERIC_MODEL_NAME — mirrors the real target-vs-judge separation
GENERIC_API_VERSION = "2025-01-01-preview"
