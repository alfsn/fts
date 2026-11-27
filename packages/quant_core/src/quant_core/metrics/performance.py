"""Return, drawdown and risk-adjusted ratios of one portfolio (Epic 2)."""

from typing import Optional

import pandas as pd

from ..analytics import (
    RiskFree,
    compute_cagr,
    compute_calmar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_volatility,
    daily_returns,
    max_time_underwater_days,
    wealth_index,
    years_between,
)
from ..models import PerformanceMetrics


def compute_performance_metrics(
    values: pd.Series,
    cash_flows: Optional[pd.Series] = None,
    risk_free_rate: RiskFree = 0.0,
) -> PerformanceMetrics:
    """
    Path metrics (return, drawdown) use the full-resolution flow-adjusted wealth
    index; dispersion ratios use its daily closes with the observed periods/year.
    ``risk_free_rate``: annual rate, constant or a Series (``rates.excess_returns``).
    Unpriced (NaN) valuations are skipped and counted in ``unpriced_obs``.
    """
    values = values.sort_index()
    unpriced = int(values.isna().sum())
    if values.count() < 2:
        return PerformanceMetrics(unpriced_obs=unpriced)

    wealth = wealth_index(values, cash_flows)
    total_return = float(wealth.iloc[-1] - 1)
    cagr = compute_cagr(total_return, years_between(wealth.index[0], wealth.index[-1]))
    max_dd = compute_max_drawdown(wealth)

    returns, ppy = daily_returns(wealth)
    excess, _ = daily_returns(wealth, risk_free_rate)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=compute_volatility(returns, ppy),
        sharpe_ratio=compute_sharpe(excess, ppy),
        sortino_ratio=compute_sortino(excess, ppy),
        calmar_ratio=compute_calmar(cagr, max_dd),
        max_drawdown=max_dd,
        time_underwater_days=max_time_underwater_days(wealth),
        unpriced_obs=unpriced,
    )
