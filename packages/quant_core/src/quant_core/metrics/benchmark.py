"""
Alpha/beta and the other benchmark-relative KPIs of one portfolio (PR 7).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..analytics import (
    RiskFree,
    align_daily,
    compute_capture_ratios,
    compute_information_ratio,
    excess_returns,
    hac_ols,
    periods_per_year,
    wealth_index,
)
from ..models import BenchmarkMetrics
from ._common import MIN_SAMPLE_OBS, _empty_frame

# A beta on fewer observations is mostly noise (PR 7, D5).
MIN_BENCHMARK_OBS = MIN_SAMPLE_OBS
CONFIDENCE = 0.95


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
    ``min_obs`` aligned returns every statistic is NaN; ``n_obs`` and ``min_obs``
    are still set.
    """
    values = values.sort_index()
    if len(values) < 2:
        return BenchmarkAnalysis(BenchmarkMetrics(min_obs=min_obs))
    aligned = align_daily(wealth_index(values, cash_flows), benchmark.sort_index())
    if len(aligned) < 2:
        return BenchmarkAnalysis(BenchmarkMetrics(min_obs=min_obs))

    wealth = aligned / aligned.iloc[0]
    # One path from levels to returns: plain returns are excess returns at 0.
    returns = pd.DataFrame({c: excess_returns(aligned[c]) for c in aligned.columns})
    excess = pd.DataFrame(
        {c: excess_returns(aligned[c], risk_free_rate) for c in aligned.columns}
    )
    excess["fitted"] = float("nan")
    n = len(returns)
    if n < min_obs:
        return BenchmarkAnalysis(
            BenchmarkMetrics(n_obs=n, min_obs=min_obs), wealth, excess
        )

    ppy = periods_per_year(aligned.index)
    up, down = compute_capture_ratios(returns["portfolio"], returns["benchmark"])
    shared = dict(
        n_obs=n,
        min_obs=min_obs,
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
