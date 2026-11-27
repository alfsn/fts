"""
Single entry point composing the analytics primitives into ``PerformanceMetrics``,
``BenchmarkMetrics`` and ``VolForecastMetrics``. Both fts backtests and the portfolio analyser go
through here so a KPI means the same thing everywhere.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .analytics import (
    EWMA_SEED_OBS,
    RISKMETRICS_DAILY_LAMBDA,
    RiskFree,
    align_daily,
    annualise_variance,
    compute_cagr,
    compute_calmar,
    compute_capture_ratios,
    compute_information_ratio,
    compute_max_drawdown,
    compute_rolling_vol,
    compute_sharpe,
    compute_sortino,
    compute_volatility,
    daily_returns,
    evaluate_variance_forecast,
    ewma_variance,
    excess_returns,
    hac_ols,
    max_time_underwater_days,
    periods_per_year,
    to_daily,
    wealth_index,
    years_between,
)
from .models import BenchmarkMetrics, PerformanceMetrics, VolForecastMetrics

# A beta on less than a year of daily data is mostly noise (PR 7, D5).
MIN_BENCHMARK_OBS = 252
CONFIDENCE = 0.95
# Forecast-quality statistics need a year of out-of-sample days, like beta.
MIN_VOL_EVAL_OBS = MIN_BENCHMARK_OBS


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

    returns, ppy = daily_returns(wealth)
    excess = excess_returns(to_daily(wealth), risk_free_rate)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=compute_volatility(returns, ppy),
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


def compute_rolling_vols(
    values: pd.Series,
    cash_flows: Optional[pd.Series],
    windows: Sequence[int],
) -> Dict[str, pd.Series]:
    """Annualised rolling volatility of the daily flow-adjusted wealth index,
    keyed by window label (``"30d"``)."""
    returns, ppy = daily_returns(wealth_index(values.sort_index(), cash_flows))
    return {f"{w}d": compute_rolling_vol(returns, w, ppy) for w in windows}


@dataclass
class VolForecastAnalysis:
    """What the charts need: metrics and the annualised volatility series."""

    metrics: VolForecastMetrics = field(default_factory=VolForecastMetrics)
    # At date t: forecast = EWMA for t+1, baseline = rolling one-year for t+1,
    # realised = volatility of the next ``horizon`` returns. Annualised fractions.
    frame: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(["forecast", "realised", "baseline"])
    )


def compute_vol_forecast(
    values: pd.Series,
    cash_flows: Optional[pd.Series] = None,
    lam: float = RISKMETRICS_DAILY_LAMBDA,
    seed_obs: int = EWMA_SEED_OBS,
    min_obs: int = MIN_VOL_EVAL_OBS,
) -> VolForecastAnalysis:
    """
    RiskMetrics EWMA volatility forecast on the daily returns of the
    flow-adjusted wealth index, checked out of sample (PR 10).

    Each day's forecast uses returns up to the day before. It is scored
    against the realised squared return next to a rolling one-year baseline
    (sample variance over ``round(periods/year)`` returns), both on the same
    days. The chart's realised line covers the next month
    (``round(periods/year / 12)`` returns). Below ``min_obs`` common days the
    scores are NaN; the counts are always set.
    """
    values = values.sort_index()
    if len(values) < 2:
        return VolForecastAnalysis(VolForecastMetrics(lam=lam))
    returns, ppy = daily_returns(wealth_index(values, cash_flows))
    if returns.empty or not ppy > 0:
        return VolForecastAnalysis(VolForecastMetrics(lam=lam, n_obs=len(returns)))
    window, horizon = max(2, round(ppy)), max(2, round(ppy / 12))

    ewma = ewma_variance(returns, lam, seed_obs)
    baseline = compute_rolling_vol(returns, window, 1.0) ** 2
    forecast_for_t, baseline_for_t = ewma.shift(1), baseline.shift(1)
    common = forecast_for_t.notna() & baseline_for_t.notna()

    def score(forecast: pd.Series):
        return evaluate_variance_forecast(
            returns[common], forecast[common], ppy, min_obs
        )

    frame = pd.DataFrame(
        {
            "forecast": annualise_variance(ewma, ppy),
            "realised": compute_rolling_vol(returns, horizon, ppy).shift(-horizon),
            "baseline": annualise_variance(baseline, ppy),
        }
    )
    return VolForecastAnalysis(
        VolForecastMetrics(
            lam=lam,
            n_obs=len(returns),
            forecast_vol=float(annualise_variance(ewma.iloc[-1], ppy)),
            baseline_window=window,
            horizon=horizon,
            ewma=score(forecast_for_t),
            baseline=score(baseline_for_t),
        ),
        frame,
    )
