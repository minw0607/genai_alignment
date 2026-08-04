"""Native mechanisms — capabilities this repo builds itself because no sibling
repo provides them.

Distinct from `adapters/`, which only wraps sibling repos' published APIs. A
module belongs here when it's real new machinery *and* general enough that more
than one scenario will want it; anything specific to a single scenario stays in
that scenario's own module (see scenarios/adversarial_inputs.py's
document-review mechanism for that pattern).
"""
