"""
Epic 2 backward-analytics charts. Pure presentation: every function receives
series already computed by ``quant_core.analytics`` and does no math itself.
"""

import math
from typing import List, Mapping, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quant_core.models import BenchmarkMetrics, DrawdownEpisode, PerformanceMetrics
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


def plot_benchmark_scatter(
    excess: pd.DataFrame, metrics: BenchmarkMetrics, benchmark_name: str
) -> go.Figure:
    """Daily excess returns, portfolio vs benchmark, with quant_core's fitted line
    (``excess`` columns: portfolio, benchmark, fitted; no line when it is all NaN)."""
    fig = go.Figure(
        go.Scatter(
            x=excess["benchmark"],
            y=excess["portfolio"],
            mode="markers",
            name="Daily excess returns",
            marker=dict(color=_PORTFOLIO_COLOR, size=4, opacity=0.5),
        )
    )
    line = excess.dropna(subset=["fitted"]).sort_values("benchmark")
    if len(line):
        fig.add_trace(
            go.Scatter(
                x=line["benchmark"],
                y=line["fitted"],
                mode="lines",
                name="OLS fit",
                line=dict(color=_BASELINE_COLOR, width=2),
            )
        )
    fig.update_layout(
        title=(
            f"Portfolio vs {benchmark_name}: alpha {_fmt(metrics.alpha, '.2%')}, "
            f"beta {_fmt(metrics.beta, '.2f')}, n = {metrics.n_obs}"
        ),
        xaxis_title=f"{benchmark_name} daily excess return",
        yaxis_title="Portfolio daily excess return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
    )
    return fig


def _estimate(value: float, se: float, low: float, high: float, pattern: str) -> str:
    if math.isnan(value):
        return "n/a"
    return (
        f"{format(value, pattern)} ± {_fmt(se, pattern)} "
        f"[{_fmt(low, pattern)}, {_fmt(high, pattern)}]"
    )


def plot_benchmark_kpi_table(
    metrics: BenchmarkMetrics, benchmark_name: str, min_obs: int
) -> go.Figure:
    m = metrics
    rows = [
        (
            "Alpha (arithmetic, ann.) ± HAC SE [95% CI]",
            _estimate(m.alpha, m.alpha_se, m.alpha_ci_low, m.alpha_ci_high, ".2%"),
        ),
        (
            "Beta ± HAC SE [95% CI]",
            _estimate(m.beta, m.beta_se, m.beta_ci_low, m.beta_ci_high, ".2f"),
        ),
        ("Dimson Beta (lag 1)", _fmt(m.dimson_beta, ".2f")),
        ("R²", _fmt(m.r_squared, ".2f")),
        ("Up Capture (daily)", _fmt(m.up_capture, ".0%")),
        ("Down Capture (daily)", _fmt(m.down_capture, ".0%")),
        ("Information Ratio", _fmt(m.information_ratio, ".2f")),
        ("Observations (daily)", f"{m.n_obs} (min {min_obs})"),
        ("Newey-West lags", "n/a" if m.hac_lags is None else str(m.hac_lags)),
    ]
    fig = go.Figure(
        go.Table(
            header=dict(values=["Benchmark KPI", "Value"], fill_color="#1a1a1a"),
            cells=dict(values=list(zip(*rows)), fill_color="#000000", align="left"),
        )
    )
    fig.update_layout(title=f"Benchmark KPIs vs {benchmark_name}")
    return fig
