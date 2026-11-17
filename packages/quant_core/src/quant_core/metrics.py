"""
Single entry point composing the analytics primitives into ``PerformanceMetrics``
and ``BenchmarkMetrics``. Both fts backtests and the portfolio analyser go
through here so a KPI means the same thing everywhere.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .analytics import (
    RiskFree,
    align_daily,
    compute_cagr,
    compute_calmar,
    compute_capture_ratios,
    compute_information_ratio,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_volatility,
    excess_returns,
    hac_ols,
    max_time_underwater_days,
    periods_per_year,
    to_daily,
    wealth_index,
    years_between,
)
from .models import BenchmarkMetrics, PerformanceMetrics

# A beta on less than a year of daily data is mostly noise (PR 7, D5).
MIN_BENCHMARK_OBS = 252
CONFIDENCE = 0.95


def compute_performance_metrics(
    values: pd.Series,
    cash_flows: Optional[pd.Series] = None,
    risk_free_rate: RiskFree = 0.0,
) -> PerformanceMetrics:
    """
    Path metrics (return, drawdown) use the full-resolution flow-adjusted wealth
    index; dispersion ratios use its daily closes with the observed periods/year.
    ``risk_free_rate``: annual rate, constant or a Series (``rates.excess_returns``).
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
    excess = excess_returns(daily, risk_free_rate)
    ppy = periods_per_year(daily.index)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=compute_volatility(daily_returns, ppy),
        sharpe_ratio=compute_sharpe(excess, ppy),
        sortino_ratio=compute_sortino(excess, ppy),
        calmar_ratio=compute_calmar(cagr, max_dd),
        max_drawdown=max_dd,
        time_underwater_days=max_time_underwater_days(wealth),
    )


def _empty_frame(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=columns, dtype=float)


@dataclass
class BenchmarkAnalysis:
    """What the charts need: metrics, aligned growth of 1, and excess returns."""

    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    # columns portfolio, benchmark; both 1.0 on the first common date
    wealth: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(["portfolio", "benchmark"])
    )
    # columns portfolio, benchmark (daily excess returns), fitted (regression line)
    excess: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(["portfolio", "benchmark", "fitted"])
    )


def compute_benchmark_metrics(
    values: pd.Series,
    cash_flows: Optional[pd.Series],
    benchmark: pd.Series,
    risk_free_rate: RiskFree = 0.0,
    min_obs: int = MIN_BENCHMARK_OBS,
) -> BenchmarkAnalysis:
    """
    Alpha/beta (OLS of daily excess returns, Newey-West SEs, t confidence
    intervals), Dimson beta (lag 1), R², capture ratios and information ratio.

    ``benchmark`` is a price level (e.g. a total-return series) in the
    portfolio's currency, on any calendar. The portfolio's flow-adjusted wealth
    index is built *before* aligning, so flows on dropped dates are kept. Below
    ``min_obs`` aligned returns every statistic is NaN; ``n_obs`` is still set.
    """
    values = values.sort_index()
    if len(values) < 2:
        return BenchmarkAnalysis()
    aligned = align_daily(wealth_index(values, cash_flows), benchmark.sort_index())
    if len(aligned) < 2:
        return BenchmarkAnalysis()

    wealth = aligned / aligned.iloc[0]
    returns = aligned.pct_change().iloc[1:]
    excess = pd.DataFrame(
        {c: excess_returns(aligned[c], risk_free_rate) for c in aligned.columns}
    )
    excess["fitted"] = float("nan")
    n = len(returns)
    if n < min_obs:
        return BenchmarkAnalysis(BenchmarkMetrics(n_obs=n), wealth, excess)

    ppy = periods_per_year(aligned.index)
    up, down = compute_capture_ratios(returns["portfolio"], returns["benchmark"])
    shared = dict(
        n_obs=n,
        r_squared=float(returns["portfolio"].corr(returns["benchmark"]) ** 2),
        up_capture=up,
        down_capture=down,
        information_ratio=compute_information_ratio(
            returns["portfolio"], returns["benchmark"], ppy
        ),
    )
    # A missing risk-free quote leaves alpha and beta undefined, not fitted on
    # a silently smaller sample.
    if excess[["portfolio", "benchmark"]].isna().any().any():
        return BenchmarkAnalysis(BenchmarkMetrics(**shared), wealth, excess)

    y, x = excess["portfolio"], excess["benchmark"]
    try:
        fit = hac_ols(y, x.to_frame("benchmark"))
        dimson = hac_ols(
            y, pd.DataFrame({"benchmark": x, "lag1": x.shift(1)}), lags=fit.lags
        )
    except (ValueError, np.linalg.LinAlgError):  # too few points, or a flat benchmark
        return BenchmarkAnalysis(BenchmarkMetrics(**shared), wealth, excess)
    t = stats.t.ppf(0.5 + CONFIDENCE / 2, fit.df)
    alpha, beta = fit.coef[0] * ppy, fit.coef[1]
    alpha_se, beta_se = fit.se[0] * ppy, fit.se[1]
    excess["fitted"] = fit.fitted
    return BenchmarkAnalysis(
        BenchmarkMetrics(
            **shared,
            hac_lags=fit.lags,
            alpha=alpha,
            alpha_se=alpha_se,
            alpha_ci_low=alpha - t * alpha_se,
            alpha_ci_high=alpha + t * alpha_se,
            beta=beta,
            beta_se=beta_se,
            beta_ci_low=beta - t * beta_se,
            beta_ci_high=beta + t * beta_se,
            dimson_beta=float(dimson.coef[1] + dimson.coef[2]),
        ),
        wealth,
        excess,
    )
