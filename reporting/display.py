"""Generic, non-identifying labels for reports and notebook output.

The real `TARGET_MODEL` / `JUDGE_MODEL` deployment names and `OPENAI_API_VERSION`
read from `.env` are the user's own confidential test configuration, not
something every reader of a report or a committed sample HTML file should see
(each user of this repo has their own). They're used as-is for actual API
calls, but must never appear in a notebook's displayed output, a saved report,
or a committed `docs/samples/*.html` file — use these generic stand-ins there
instead.

`GENERIC_PROVIDER_NAME` exists for the same reason: this repo's own dev/test
setup happens to run against Azure OpenAI, but every scenario's client code
only assumes an OpenAI-compatible interface (`OPENAI_API_KEY`/`OPENAI_BASE_URL`/
`OPENAI_API_VERSION`) — a different user of this repo may point it at plain
OpenAI, a different provider entirely behind an OpenAI-compatible proxy, or
something else. Hardcoding "Azure OpenAI" into a report's Target line or
Provider field would be both wrong for them and an unintended hint about
this repo's own specific setup — never hardcode it; use this instead.
"""

GENERIC_MODEL_NAME = "gpt-5.5"
GENERIC_JUDGE_MODEL_NAME = "gpt-5.4"  # distinct from GENERIC_MODEL_NAME — mirrors the real target-vs-judge separation
GENERIC_API_VERSION = "2025-01-01-preview"
GENERIC_PROVIDER_NAME = "OpenAI-compatible API"
