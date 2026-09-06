"""Exports that let someone outside this repo use what is inside it.

The standing weakness of use-case-grounded evaluation is that nobody else can
run it. A public benchmark is portable by construction — its cases are the
artefact. Hand-authored scenarios are not: the cases live in a fixture, the
system under test lives in a local harness, and the scoring lives in a module
that knows both. Reproduction outside this repository is possible in principle
and inconvenient enough in practice that it does not happen.

This package narrows that gap by writing the two portable halves — **the cases**
and **the results** — into shapes another tool already reads.
"""
