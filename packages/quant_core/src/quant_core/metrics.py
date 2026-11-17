"""
Single entry point composing the analytics primitives into ``PerformanceMetrics``.
Both fts backtests and the portfolio analyser go through here so a KPI means
the same thing everywhere.
"""

from typing import Optional

import pandas as pd

from .analytics import (
    compute_cagr,
    compute_calmar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_volatility,
    max_time_underwater_days,
    periods_per_year,
    to_daily,
    wealth_index,
    years_between,
)
from .models import PerformanceMetrics


def compute_performance_metrics(
    values: pd.Series,
    cash_flows: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Path metrics (return, drawdown) use the full-resolution flow-adjusted wealth
    index; dispersion ratios use its daily closes with the observed periods/year.
    """
    values = values.sort_index()
    if len(values) < 2:
        return PerformanceMetrics()

    wealth = wealth_index(values, cash_flows)
    total_return = float(wealth.iloc[-1] - 1)
    cagr = compute_cagr(total_return, years_between(values.index[0], values.index[-1]))
    max_dd = compute_max_drawdown(wealth)

    daily = to_daily(wealth)
    daily_returns = daily.pct_change().dropna()
    ppy = periods_per_year(daily.index)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=compute_volatility(daily_returns, ppy),
        sharpe_ratio=compute_sharpe(daily_returns, ppy, risk_free_rate),
        sortino_ratio=compute_sortino(daily_returns, ppy, risk_free_rate),
        calmar_ratio=compute_calmar(cagr, max_dd),
        max_drawdown=max_dd,
        time_underwater_days=max_time_underwater_days(wealth),
    )
