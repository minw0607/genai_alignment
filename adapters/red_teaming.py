"""Adapter onto llm_red_teaming's prompt-injection workstream.

Like multi_agent_otel_eval, this sibling repo has no pyproject.toml — it's
not pip-installable, only clonable. This adapter references a local sibling
checkout via sys.path instead of a pip dependency (see README Setup): clone
it alongside this repo's parent directory —

    git clone https://github.com/minw0607/llm_red_teaming llm_red_teaming

Calls its stable PromptInjectionRunner, STRATEGIES taxonomy, and
AzureOpenAITarget directly — attack-strategy transforms and canary
detection stay in llm_red_teaming. No evaluation logic is duplicated here.

This adapter is used two ways by scenarios/adversarial_inputs.py:
- The retail-banking chatbot track passes its own `tasks` dict (banking
  queries, not llm_red_teaming's generic translate/summarize/sentiment) to
  PromptInjectionRunner, reusing its strategies and canary detection
  unchanged — genuine reuse, just pointed at a different target task.
- The financial-document-review track doesn't fit PromptInjectionRunner at
  all (hidden-text-in-a-document isn't one of the Open-Prompt-Injection
  strategies, and scoring is a decision-outcome comparison, not canary
  detection) — that track only needs AzureOpenAITarget for connectivity,
  exposed here via build_target()/build_judge(), with the actual test
  logic native to genai_alignment in scenarios/adversarial_inputs.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

SIBLING_REPO = Path(__file__).resolve().parent.parent.parent / "llm_red_teaming"


def _ensure_sibling_on_path() -> None:
    if not SIBLING_REPO.exists():
        raise FileNotFoundError(
            f"llm_red_teaming not found at {SIBLING_REPO} — clone it as a sibling "
            "of this repo's parent directory: "
            "git clone https://github.com/minw0607/llm_red_teaming llm_red_teaming"
        )
    path = str(SIBLING_REPO)
    if path not in sys.path:
        sys.path.insert(0, path)


class PromptInjectionHarness:
    """Build once per notebook, then call run_canary_benchmark() /
    build_target() / build_judge() as many times as needed.

    The target model reads TARGET_MODEL; the judge reads JUDGE_MODEL
    (falling back to TARGET_MODEL if unset) — same repo-wide independent-
    judge convention as scenarios 1-3. AzureOpenAITarget already reads this
    repo's exact env var convention (OPENAI_API_KEY/OPENAI_BASE_URL/
    OPENAI_API_VERSION), so no wrapper is needed beyond passing which model
    each role should use.
    """

    def __init__(self, target_model: str, tasks: dict | None = None):
        _ensure_sibling_on_path()

        from attacks.prompt import PromptInjectionRunner
        from targets.azure_openai import AzureOpenAITarget

        self.target = AzureOpenAITarget(model=target_model)
        self.judge = AzureOpenAITarget(model=os.environ.get("JUDGE_MODEL", target_model))
        self.runner = PromptInjectionRunner(self.target, tasks=tasks)

    def build_target(self) -> "object":
        return self.target

    def build_judge(self) -> "object":
        return self.judge

    def run_canary_benchmark(
        self, context: str = "direct", n_per_task: int = 2, checkpoint_path: str | None = None,
    ) -> pd.DataFrame:
        """context: 'direct' (injection in user input) or 'indirect' (injection
        hidden in a document the model must process, using llm_red_teaming's
        own generic document-QA framing — see scenarios/adversarial_inputs.py
        for why the financial-document-review track uses a native mechanism
        instead of this for its indirect testing)."""
        results = self.runner.run(context=context, n_per_task=n_per_task, checkpoint_path=checkpoint_path)
        return pd.DataFrame([asdict(r) for r in results])
