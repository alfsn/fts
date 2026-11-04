"""
Epic 2 backward-analytics charts. Pure presentation: every function receives
series already computed by ``quant_core.analytics`` and does no math itself.
"""

import math
from typing import List, Mapping, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quant_core.models import DrawdownEpisode, PerformanceMetrics
from quant_ui.plots.theme import BBG_DOWN, BBG_TEXT, setup_bloomberg_theme

setup_bloomberg_theme()

_PORTFOLIO_COLOR = "#2196f3"
_BASELINE_COLOR = "#78909c"


def plot_cumulative_returns(
    cumulative: pd.Series,
    baseline: Optional[pd.Series] = None,
    baseline_name: str = "Baseline",
) -> go.Figure:
    """Cumulative (compounded) return vs an optional baseline, both as fractions."""
    fig = go.Figure(
        go.Scatter(
            x=cumulative.index,
            y=cumulative,
            mode="lines",
            name="Portfolio",
            line=dict(color=_PORTFOLIO_COLOR, width=2),
        )
    )
    if baseline is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline.index,
                y=baseline,
                mode="lines",
                name=baseline_name,
                line=dict(color=_BASELINE_COLOR, width=1.5, dash="dash"),
            )
        )
    fig.update_layout(title="Cumulative Return (TWR)", yaxis_tickformat=".1%")
    return fig


def plot_rolling_volatility(rolling_vols: Mapping[str, pd.Series]) -> go.Figure:
    """One sub-chart per window label, e.g. {"30d": ..., "90d": ..., "252d": ...}."""
    fig = make_subplots(
        rows=len(rolling_vols),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{label} Rolling Volatility" for label in rolling_vols],
    )
    for row, (label, vol) in enumerate(rolling_vols.items(), start=1):
        fig.add_trace(
            go.Scatter(x=vol.index, y=vol, mode="lines", name=label), row=row, col=1
        )
        fig.update_yaxes(tickformat=".0%", row=row, col=1)
    fig.update_layout(height=250 * len(rolling_vols), showlegend=False)
    return fig


def plot_underwater(
    drawdowns: pd.Series, top_episodes: List[DrawdownEpisode]
) -> go.Figure:
    """Red-shaded underwater curve with the given top episodes highlighted."""
    fig = go.Figure(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color=BBG_DOWN, width=1),
        )
    )
    last = drawdowns.index[-1] if len(drawdowns) else None
    for rank, ep in enumerate(top_episodes, start=1):
        fig.add_vrect(
            x0=ep.peak,
            x1=ep.recovery or last,
            fillcolor=BBG_DOWN,
            opacity=0.12,
            line_width=0,
            annotation_text=f"#{rank} {ep.depth:.1%}",
            annotation_position="bottom left",
            annotation_font_color=BBG_TEXT,
        )
    fig.update_layout(title="Underwater Plot", yaxis_tickformat=".0%")
    return fig


def _fmt(value: float, pattern: str) -> str:
    return "n/a" if math.isnan(value) else format(value, pattern)


def plot_risk_kpi_table(metrics: PerformanceMetrics) -> go.Figure:
    rows = [
        ("Total Return (TWR)", _fmt(metrics.total_return, ".2%")),
        ("CAGR", _fmt(metrics.cagr, ".2%")),
        ("Volatility (ann.)", _fmt(metrics.volatility, ".2%")),
        ("Sharpe Ratio", _fmt(metrics.sharpe_ratio, ".2f")),
        ("Sortino Ratio", _fmt(metrics.sortino_ratio, ".2f")),
        ("Calmar Ratio", _fmt(metrics.calmar_ratio, ".2f")),
        ("Max Drawdown", _fmt(metrics.max_drawdown, ".2%")),
        ("Longest Time Underwater", _fmt(metrics.time_underwater_days, ".0f") + " d"),
    ]
    fig = go.Figure(
        go.Table(
            header=dict(values=["Risk KPI", "Value"], fill_color="#1a1a1a"),
            cells=dict(values=list(zip(*rows)), fill_color="#000000", align="left"),
        )
    )
    fig.update_layout(title="Risk KPIs")
    return fig
