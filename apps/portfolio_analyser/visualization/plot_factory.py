"""Composes quant_core analytics with quant_ui charts for the backward-analytics report (Epic 2)."""

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from quant_core.analytics import (
    compute_rolling_vol,
    drawdown_series,
    periods_per_year,
    to_daily,
    top_drawdowns,
    wealth_index,
)
from quant_core.metrics import compute_performance_metrics
from quant_ui.plots.analytics import (
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
    baseline: Optional[pd.Series] = None,
    baseline_name: str = "Baseline",
) -> List[go.Figure]:
    """``values``: portfolio valuations; ``baseline``: benchmark prices on any index."""
    wealth = wealth_index(values, cash_flows)
    daily_returns = to_daily(wealth).pct_change().dropna()
    ppy = periods_per_year(daily_returns.index)

    return [
        plot_risk_kpi_table(compute_performance_metrics(values, cash_flows)),
        plot_cumulative_returns(
            wealth - 1,
            baseline=None if baseline is None else wealth_index(baseline) - 1,
            baseline_name=baseline_name,
        ),
        plot_rolling_volatility(
            {
                f"{w}d": compute_rolling_vol(daily_returns, w, ppy)
                for w in ROLLING_WINDOWS
            }
        ),
        plot_underwater(drawdown_series(wealth), top_drawdowns(wealth, TOP_DRAWDOWNS)),
    ]


def write_report(figures: List[go.Figure], path: str) -> None:
    body = "".join(
        fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        for i, fig in enumerate(figures)
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<html><body style='background:#000'>{body}</body></html>")
