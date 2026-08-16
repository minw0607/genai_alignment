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


# ---------------------------------------------------------------- Scrubbing
# The constants above only help where *we* write the provider's name. They do
# nothing about the other route it takes into a report: **raw API error text
# stored alongside a result and later rendered.** A gateway refusal arrives as
# a message naming the provider, gets saved in a `response` or `error` column,
# and is published the moment that column reaches a results table — past every
# generic label, because no scenario code ever typed the name.
#
# That is not hypothetical: it put the provider's name into a committed sample
# report 14 times while every Provider field on the same page read correctly.
#
# Scrub at the boundary where stored text becomes rendered text.

import re as _re

#: (pattern, replacement) applied in order. Each keeps the *diagnostic* content
#: — that a platform filter refused the call, and why — while dropping the
#: vendor identity, so the row still explains itself.
_PROVIDER_SCRUBS: list[tuple[str, str]] = [
    (r"Azure OpenAI's content management policy", "the platform's content management policy"),
    (r"Azure OpenAI Service", "the platform"),
    (r"\bAzure OpenAI\b", "the platform"),
    (r"\bAzure\b", "the platform"),
    (r"https?://go\.microsoft\.com/\S*", "(provider documentation)"),
    (r"https?://\S*\.openai\.azure\.com\S*", "(endpoint)"),
    (r"https?://chatgpt\.com/\S*", "(provider documentation)"),
    (r"\bapi[-_]version=[0-9\-]+(?:-preview)?", "api-version=(redacted)"),
]


def scrub_provider_text(value):
    """Remove provider-identifying strings from text that will be published.

    Non-strings pass through untouched, so this is safe to map over a column
    holding NaN or numbers.
    """
    if not isinstance(value, str):
        return value
    out = value
    for pattern, replacement in _PROVIDER_SCRUBS:
        out = _re.sub(pattern, replacement, out)
    return out


def scrub_frame(df, columns=None):
    """Return a copy of `df` with free-text columns scrubbed for publication.

    Defaults to every object-dtype column, because the leak is precisely the
    column nobody remembered was free text.
    """
    out = df.copy()
    targets = columns if columns is not None else [
        c for c in out.columns if out[c].dtype == object
    ]
    for col in targets:
        if col in out.columns:
            out[col] = out[col].map(scrub_provider_text)
    return out
