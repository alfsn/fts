"""Rolling volatility and the EWMA volatility forecast with its check (PR 10)."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import pandas as pd

from ..analytics import (
    EWMA_SEED_OBS,
    RISKMETRICS_DAILY_LAMBDA,
    annualise_variance,
    compute_rolling_vol,
    daily_returns,
    evaluate_variance_forecast,
    ewma_variance,
    rolling_variance,
    wealth_index,
)
from ..models import VolForecastMetrics
from ._common import MIN_SAMPLE_OBS, _empty_frame

# Forecast scores need as many out-of-sample days as a beta needs.
MIN_VOL_EVAL_OBS = MIN_SAMPLE_OBS


def compute_rolling_vols(
    values: pd.Series,
    cash_flows: Optional[pd.Series],
    windows: Sequence[int],
) -> Dict[str, pd.Series]:
    """Annualised rolling volatility of the daily flow-adjusted wealth index,
    keyed by window label (``"30d"``: observations, not calendar days)."""
    returns, ppy = daily_returns(wealth_index(values.sort_index(), cash_flows))
    return {f"{w}d": compute_rolling_vol(returns, w, ppy) for w in windows}


@dataclass
class VolForecastAnalysis:
    """What the charts need: metrics and the annualised volatility series."""

    metrics: VolForecastMetrics = field(default_factory=VolForecastMetrics)
    # At date t: forecast = EWMA for t+1, baseline = rolling one-year for t+1,
    # realised = volatility of the next ``horizon`` returns. All zero-mean,
    # annualised fractions.
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
    (mean r² over ``round(periods/year)`` returns), both on the same days.
    The chart's realised line covers the next month (``round(periods/year /
    12)`` returns). All three are zero-mean, like RiskMetrics, so they
    estimate the same thing. Below ``min_obs`` common days the scores are
    NaN; the counts and ``min_obs`` are always set.
    """
    values = values.sort_index()
    if len(values) < 2:
        return VolForecastAnalysis(VolForecastMetrics(lam=lam, min_obs=min_obs))
    returns, ppy = daily_returns(wealth_index(values, cash_flows))
    if returns.empty or not ppy > 0:
        return VolForecastAnalysis(
            VolForecastMetrics(lam=lam, min_obs=min_obs, n_obs=len(returns))
        )
    window, horizon = max(2, round(ppy)), max(2, round(ppy / 12))

    ewma = ewma_variance(returns, lam, seed_obs)
    baseline = rolling_variance(returns, window, zero_mean=True)
    forecast_for_t, baseline_for_t = ewma.shift(1), baseline.shift(1)
    common = forecast_for_t.notna() & baseline_for_t.notna()

    def score(forecast: pd.Series):
        return evaluate_variance_forecast(
            returns[common], forecast[common], ppy, min_obs
        )

    realised = rolling_variance(returns, horizon, zero_mean=True).shift(-horizon)
    frame = pd.DataFrame(
        {
            "forecast": annualise_variance(ewma, ppy),
            "realised": annualise_variance(realised, ppy),
            "baseline": annualise_variance(baseline, ppy),
        }
    )
    return VolForecastAnalysis(
        VolForecastMetrics(
            lam=lam,
            min_obs=min_obs,
            n_obs=len(returns),
            forecast_vol=float(annualise_variance(ewma.iloc[-1], ppy)),
            baseline_window=window,
            horizon=horizon,
            ewma=score(forecast_for_t),
            baseline=score(baseline_for_t),
        ),
        frame,
    )
