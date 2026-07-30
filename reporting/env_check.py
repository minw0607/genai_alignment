"""Environment self-check — run this first in every notebook.

Doesn't install or clone anything itself (that stays a deliberate, visible
step the user takes — see README Setup); it only verifies what's already
there and reports it clearly. The point is turning "ModuleNotFoundError
three cells later" or a raw stack trace from a missing sibling clone into
one readable status table at the top of the notebook, before any real work
(or API spend) starts.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path


def check_environment(
    *,
    required_packages: list[str] | None = None,
    required_env_vars: list[str] | None = None,
    sibling_repos: dict[str, str] | None = None,
) -> bool:
    """Print a pass/fail status table for this notebook's dependencies.

    Parameters
    ----------
    required_packages : importable module names (not pip package names —
        e.g. "genai_capability_bench", not the package's PyPI name).
    required_env_vars : names that must be set (and non-empty) in the
        current environment (after load_dotenv()).
    sibling_repos : {relative_path: clone_url} for non-pip-installable
        sibling repos this notebook needs (e.g.
        {"../Agent": "https://github.com/minw0607/multi_agent_otel_eval"}).

    Returns
    -------
    True if every check passed, False otherwise — check this if you want
    the notebook to stop cleanly here rather than fail deep in a later cell.
    """
    rows: list[tuple[str, str]] = []
    ok = True

    for pkg in required_packages or []:
        try:
            importlib.import_module(pkg)
            rows.append((f"package: {pkg}", "OK"))
        except ImportError:
            rows.append((f"package: {pkg}", "MISSING — see README Setup (pip install -e .)"))
            ok = False

    for var in required_env_vars or []:
        if os.environ.get(var):
            rows.append((f"env var: {var}", "OK"))
        else:
            rows.append((f"env var: {var}", "MISSING — check your .env (see .env.example)"))
            ok = False

    for path, url in (sibling_repos or {}).items():
        if Path(path).exists():
            rows.append((f"sibling repo: {path}", "OK"))
        else:
            rows.append((f"sibling repo: {path}", f"MISSING — git clone {url} {path}"))
            ok = False

    width = max((len(r[0]) for r in rows), default=0)
    for label, status in rows:
        marker = "✓" if status == "OK" else "✗"
        print(f"{marker} {label.ljust(width)}  {status}")

    print("\nAll checks passed." if ok else "\nFix the items above before continuing.")
    return ok
