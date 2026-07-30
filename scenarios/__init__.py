"""Per-scenario logic — one module per scenario, imported by that scenario's notebook.

Fixtures (scenarios/fixtures/*.jsonl) are the data; the modules here are the
code specific to one scenario (data loading, charts, report assembly).
Anything generic across scenarios (multi-dataset combination, judge review,
the HTML report template) lives in reporting/, not here.
"""
