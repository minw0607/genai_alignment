"""Adapter onto llm_red_teaming's prompt-injection workstream.

Like multi_agent_otel_eval, this sibling repo has no pyproject.toml — it's
not pip-installable, only clonable. This adapter references a local sibling
checkout via sys.path instead of a pip dependency (see README Setup): clone
it alongside this repo's parent directory —

    git clone https://github.com/minw0607/llm_red_teaming llm_red_teaming

Calls its stable PromptInjectionRunner and AzureOpenAITarget directly —
attack strategies, canary detection, and the real-payload LLM judge all stay
in llm_red_teaming. No evaluation logic is duplicated here.
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
    """Build once per notebook, then call run_canary_benchmark() / run_payload_track()
    as many times as needed.

    The target model reads TARGET_MODEL; the judge for the real-payload track
    reads JUDGE_MODEL (falling back to TARGET_MODEL if unset) — same
    repo-wide independent-judge convention as scenarios 1-3.
    AzureOpenAITarget already reads this repo's exact env var convention
    (OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_API_VERSION), so no wrapper is
    needed beyond passing which model each role should use.
    """

    def __init__(self, target_model: str):
        _ensure_sibling_on_path()

        from attacks.prompt import PromptInjectionRunner, load_injection_payloads
        from targets.azure_openai import AzureOpenAITarget

        self._load_injection_payloads = load_injection_payloads
        self.target = AzureOpenAITarget(model=target_model)
        self.judge = AzureOpenAITarget(model=os.environ.get("JUDGE_MODEL", target_model))
        self.runner = PromptInjectionRunner(self.target)

    def run_canary_benchmark(
        self, context: str, n_per_task: int = 2, checkpoint_path: str | None = None,
    ) -> pd.DataFrame:
        """context: 'direct' (injection in user input) or 'indirect' (injection
        hidden in a document the model must process). Runs all 5 strategies
        across all 3 base tasks."""
        results = self.runner.run(context=context, n_per_task=n_per_task, checkpoint_path=checkpoint_path)
        return pd.DataFrame([asdict(r) for r in results])

    def run_payload_track(self, max_items: int = 30, checkpoint_path: str | None = None) -> pd.DataFrame:
        """Real-world injection strings (deepset/prompt-injections), judged by
        JUDGE_MODEL rather than canary detection, since these freeform
        payloads carry no canary marker."""
        payloads = self._load_injection_payloads()
        results = self.runner.run_payloads(
            payloads, judge_target=self.judge, max_items=max_items, checkpoint_path=checkpoint_path,
        )
        return pd.DataFrame([asdict(r) for r in results])
