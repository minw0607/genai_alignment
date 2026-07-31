"""Adapter onto multi_agent_otel_eval.

Unlike genai_capability_bench, this sibling repo has no pyproject.toml — it's
not pip-installable, only clonable. This adapter references a local sibling
checkout via sys.path instead of a pip dependency (see README Setup): clone
it alongside this repo's parent directory —

    git clone https://github.com/minw0607/multi_agent_otel_eval Agent

Calls its stable evaluate_batch() API only — agent orchestration, tool
scoring, and tracing all stay in multi_agent_otel_eval.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

SIBLING_REPO = Path(__file__).resolve().parent.parent.parent / "Agent"


def _ensure_sibling_on_path() -> None:
    if not SIBLING_REPO.exists():
        raise FileNotFoundError(
            f"multi_agent_otel_eval not found at {SIBLING_REPO} — clone it as a sibling "
            "of this repo's parent directory: "
            "git clone https://github.com/minw0607/multi_agent_otel_eval Agent"
        )
    path = str(SIBLING_REPO)
    if path not in sys.path:
        sys.path.insert(0, path)


def _configure_env(target_model: str) -> None:
    """Point the sibling repo's agent role at our target model. JUDGE_MODEL is
    left to come from .env (a genuinely different model from the one under
    test, per repo convention — see .env.example) and only falls back to
    target_model here if .env doesn't set it. Also fixes a reasoning-model
    quirk: this model rejects any temperature other than its default (1) —
    multi_agent_otel_eval's LLM factory always passes an explicit temperature,
    unlike the raw-completion clients elsewhere in this repo that can omit it
    entirely, so the workaround here is forcing it to 1 rather than omitting
    it."""
    os.environ.setdefault("AGENT_MODEL", target_model)
    os.environ.setdefault("JUDGE_MODEL", target_model)
    os.environ.setdefault("AGENT_TEMPERATURE", "1")
    os.environ.setdefault("JUDGE_TEMPERATURE", "1")
    os.environ.setdefault("DATA_DIR", str(SIBLING_REPO / "outputs" / "data"))
    os.environ.setdefault("OUTPUT_DIR", "outputs/agent_runs")
    os.environ.setdefault("TRACE_DIR", "outputs/agent_runs/traces")


class AgentHarness:
    """Build once per notebook, then call `run()` as many times as needed.

    Reuses the same tracer/evaluator/agent objects across repeated calls —
    matching how the sibling repo's own demo notebook uses them, and how
    evaluate_batch() expects them (it resets tracer/health-monitor state
    internally on every call, so reusing the objects across repeats is safe
    and is what the upstream code is designed for).
    """

    def __init__(self, target_model: str):
        _ensure_sibling_on_path()
        _configure_env(target_model)

        from src import (
            Config,
            CostTracker,
            HealthMonitor,
            HierarchicalTracer,
            HybridEvaluator,
            TracingManager,
            create_baseline_agent,
            create_multi_agent,
            evaluate_batch,
            load_mind2web,
        )

        Config.setup_dirs()
        self._evaluate_batch = evaluate_batch
        self._load_mind2web = load_mind2web
        self.config = Config

        self.tracing_manager = TracingManager()
        self.hier_tracer = HierarchicalTracer()
        self.cost_tracker = CostTracker()
        self.health_monitor = HealthMonitor(window_size=50)

        self.agent, judge_llm = create_baseline_agent(Config)
        self.multi_agents = create_multi_agent(Config)
        self.evaluator = HybridEvaluator(
            judge_llm,
            pass_threshold=Config.EVAL_PASS_THRESHOLD,
            rule_weight=Config.RULE_WEIGHT,
            llm_weight=Config.LLM_WEIGHT,
        )

    def load_tasks(self, n: int) -> list[dict]:
        """Load a fixed set of the first n Mind2Web tasks from the cached dataset."""
        tasks = self._load_mind2web(self.config.DATA_DIR, target_tasks=max(n, 20))
        return tasks[:n]

    def run(self, tasks: list[dict], mode: str, verbose: bool = False) -> pd.DataFrame:
        """Run all of `tasks` once through the given mode ("single" or "multi")."""
        kwargs = dict(
            evaluator=self.evaluator,
            hier_tracer=self.hier_tracer,
            tracing_manager=self.tracing_manager,
            cost_tracker=self.cost_tracker,
            health_monitor=self.health_monitor,
            config=self.config,
            verbose=verbose,
        )
        if mode == "single":
            kwargs["agent"] = self.agent
        elif mode == "multi":
            kwargs["multi_agents"] = self.multi_agents
        else:
            raise ValueError(f"mode must be 'single' or 'multi', got {mode!r}")

        results, _ = self._evaluate_batch(tasks, len(tasks), mode, **kwargs)
        return results
