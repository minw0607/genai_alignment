"""Fixture generation — growing hand-authored test cases into larger sets.

Every scenario in this library is built on **use-case fixtures**: records and
requests written by hand to look like the work a real system does, rather than
items drawn from a public benchmark. That choice buys realism and costs sample
size, and the cost is not cosmetic — at six cases per arm a Fisher test cannot
return a p-value below 0.10 no matter how clean the separation is. Several
findings in this repo are reported as *underpowered, not null* for exactly that
reason.

This package closes that gap without abandoning the hand-authored design: the
seed cases become **few-shot templates**, a model proposes more of them, and
every proposal passes structural gates before it is allowed into a fixture.
The approach follows Anthropic's Bloom (Dec 2025) — behaviour spec, ideate,
validate, judge — with one deliberate difference noted in `bloom.py`.
"""
