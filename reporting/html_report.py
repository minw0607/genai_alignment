"""Uniform HTML testing-report template, shared across every scenario.

Every scenario notebook builds a `ScenarioReport` from its own run data and
calls `render_report` — the template (reporting/templates/scenario_report.html.j2)
is fixed so reports read the same way scenario to scenario, while
`extra_sections` gives each scenario room for whatever doesn't fit the
standard shape (see docs/drift_detection.md for an example of scenario-specific
content: noise-floor calibration, controlled-drift validation).
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

TEMPLATE_DIR = Path(__file__).parent / "templates"

TIER_COLORS = {
    "Tier 1": "#2a9d8f",
    "Tier 2": "#e9762b",
    "Tier 3": "#c1121f",
}


def fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG string for embedding inline."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _inline_md(text: str) -> Markup:
    """Escape text, then re-enable a small safe subset of markdown (**bold**, `code`)."""
    out = escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return Markup(out)


@dataclass
class DataSection:
    name: str
    layer: str  # e.g. "Layer 6 — custom-authored"
    source: str
    size: str
    description: str


@dataclass
class Metric:
    label: str
    value: str
    sublabel: str = ""


@dataclass
class ChartImage:
    title: str
    caption: str
    base64_png: str
    section: str = "results"  # "data" | "results" | "other"


@dataclass
class ExtraSection:
    title: str
    html: str


@dataclass
class ScenarioReport:
    scenario_name: str
    tier: str
    risk: str
    goal: str
    target_summary: dict[str, str]
    approach: str
    data_sections: list[DataSection]
    key_metrics: list[Metric]
    results_tables: list[tuple[str, pd.DataFrame]]
    charts: list[ChartImage] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    extra_sections: list[ExtraSection] = field(default_factory=list)
    notebook_link: str = ""
    doc_link: str = ""
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )


def render_report(report: ScenarioReport) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("scenario_report.html.j2")

    results_tables_html = [
        (title, df.to_html(index=False, classes="report-table", border=0, escape=True))
        for title, df in report.results_tables
    ]
    data_charts = [c for c in report.charts if c.section == "data"]
    results_charts = [c for c in report.charts if c.section == "results"]
    other_charts = [c for c in report.charts if c.section == "other"]

    return template.render(
        report=report,
        tier_color=TIER_COLORS.get(report.tier, "#495057"),
        results_tables_html=results_tables_html,
        data_charts=data_charts,
        results_charts=results_charts,
        other_charts=other_charts,
        observations_html=[_inline_md(o) for o in report.observations],
        next_steps_html=[_inline_md(o) for o in report.next_steps],
        approach_html=_inline_md(report.approach),
    )


def save_report(html: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
