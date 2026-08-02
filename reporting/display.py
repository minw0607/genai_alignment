"""Generic, non-identifying labels for reports and notebook output.

The real `TARGET_MODEL` / `JUDGE_MODEL` deployment names read from `.env`
are the user's own confidential test configuration, not something every
reader of a report or a committed sample HTML file should see (each user
of this repo has their own). They're used as-is for actual API calls, but
must never appear in a notebook's displayed output, a saved report, or a
committed `docs/samples/*.html` file — use these generic stand-ins there
instead.

`GENERIC_PROVIDER_NAME` exists for the same reason: this repo's own dev/test
setup happens to run against Azure OpenAI, but every scenario's client code
only assumes an OpenAI-compatible interface — a different user of this repo
may point it at plain OpenAI, a different provider entirely behind an
OpenAI-compatible proxy, or something else. Hardcoding "Azure OpenAI" into
a report's Target line or Provider field would be both wrong for them and
an unintended hint about this repo's own specific setup — never hardcode
it; use this instead.

Deliberately no `GENERIC_API_VERSION` here (removed 2026-08-02): "API
version" is an Azure-OpenAI-specific concept (the `api-version` query
parameter Azure's REST API requires) with no equivalent for plain OpenAI,
Anthropic, Google, or most other providers — displaying it at all, generic
value or not, implied every target was Azure-shaped. Don't re-add it as a
generic display field; if a scenario ever needs to show a provider-specific
detail like this, it belongs in that scenario's own extra_sections, not the
shared Testing Scope table every scenario renders through.
"""

GENERIC_MODEL_NAME = "gpt-5.5"
GENERIC_JUDGE_MODEL_NAME = "gpt-5.4"  # distinct from GENERIC_MODEL_NAME — mirrors the real target-vs-judge separation
GENERIC_PROVIDER_NAME = "OpenAI-compatible API"
