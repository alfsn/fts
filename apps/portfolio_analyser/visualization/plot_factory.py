"""Composes quant_core analytics with quant_ui charts for the backward-analytics
report (Epic 2, PR 7)."""

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from quant_core.analytics import (
    RiskFree,
    compute_rolling_vol,
    drawdown_series,
    periods_per_year,
    to_daily,
    top_drawdowns,
    wealth_index,
)
from quant_core.metrics import (
    MIN_BENCHMARK_OBS,
    compute_benchmark_metrics,
    compute_performance_metrics,
)
from quant_ui.plots.analytics import (
    plot_benchmark_kpi_table,
    plot_benchmark_scatter,
    plot_cumulative_returns,
    plot_risk_kpi_table,
    plot_rolling_volatility,
    plot_underwater,
)

ROLLING_WINDOWS = (30, 90, 252)
TOP_DRAWDOWNS = 5


def backward_analytics_figures(
    values: pd.Series,
    cash_flows: Optional[pd.Series] = None,
    benchmark: Optional[pd.Series] = None,
    benchmark_name: str = "Benchmark",
    risk_free_rate: RiskFree = 0.0,
) -> List[go.Figure]:
    """
    ``values``/``cash_flows``: the ledger replay. ``benchmark``: a total-return
    level in the same currency, on any calendar: quant_core aligns it to the
    portfolio's dates before any return is taken.
    """
    wealth = wealth_index(values, cash_flows)
    daily = to_daily(wealth)
    daily_returns = daily.pct_change().dropna()
    ppy = periods_per_year(daily.index)

    figures = [
        plot_risk_kpi_table(
            compute_performance_metrics(values, cash_flows, risk_free_rate)
        )
    ]
    analysis = (
        None
        if benchmark is None
        else compute_benchmark_metrics(values, cash_flows, benchmark, risk_free_rate)
    )
    if analysis is None or analysis.wealth.empty:
        figures.append(plot_cumulative_returns(wealth - 1))
    else:
        figures.append(
            plot_cumulative_returns(
                analysis.wealth["portfolio"] - 1,
                baseline=analysis.wealth["benchmark"] - 1,
                baseline_name=benchmark_name,
            )
        )
    if analysis is not None:
        figures += [
            plot_benchmark_kpi_table(
                analysis.metrics, benchmark_name, MIN_BENCHMARK_OBS
            ),
            plot_benchmark_scatter(analysis.excess, analysis.metrics, benchmark_name),
        ]
    figures += [
        plot_rolling_volatility(
            {
                f"{w}d": compute_rolling_vol(daily_returns, w, ppy)
                for w in ROLLING_WINDOWS
            }
        ),
        plot_underwater(drawdown_series(wealth), top_drawdowns(wealth, TOP_DRAWDOWNS)),
    ]
    return figures


def write_report(figures: List[go.Figure], path: str) -> None:
    body = "".join(
        fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        for i, fig in enumerate(figures)
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<html><body style='background:#000'>{body}</body></html>")
