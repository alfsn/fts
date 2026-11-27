"""
Volatility forecasting (PR 10): RiskMetrics EWMA and its out-of-sample check.

Convention: a variance *forecast* Series holds, at date t, the forecast for
the next period made with returns up to and including t. Shift it by one to
line it up with the return it forecasts (``forecast.shift(1)`` vs ``r``).
Variances are per period; annualise with ``annualise_variance`` and the
observed ``periods_per_year``.
"""

from typing import Union

import numpy as np
import pandas as pd

from ..models import ForecastEvaluation
from .regression import hac_ols

# RiskMetrics (J.P. Morgan, 1996) decay for daily data: memory ~ 1/(1-λ) ≈ 17 days.
RISKMETRICS_DAILY_LAMBDA = 0.94
# The seed variance is the mean r² of the first returns (~2x the EWMA memory).
EWMA_SEED_OBS = 30


def annualise_variance(
    variance: Union[float, pd.Series], periods_per_year: float
) -> Union[float, pd.Series]:
    """Per-period variance -> annualised volatility, sqrt(var · periods/year)."""
    return np.sqrt(variance * periods_per_year)


def ewma_variance(
    returns: pd.Series,
    lam: float = RISKMETRICS_DAILY_LAMBDA,
    seed_obs: int = EWMA_SEED_OBS,
) -> pd.Series:
    """
    Zero-mean EWMA variance: s_t = lam · s_{t-1} + (1 - lam) · r_t².
    ``s_t`` is the forecast for t+1 (see module docstring). Seeded with the
    mean r² of the first ``seed_obs`` returns, at the seed's last date; NaN
    before it and everywhere if there are fewer returns than that.
    """
    if not 0 < lam < 1:
        raise ValueError(f"lam must be in (0, 1), got {lam}.")
    if seed_obs < 1:
        raise ValueError(f"seed_obs must be >= 1, got {seed_obs}.")
    squared = returns.astype(float) ** 2
    if len(squared) < seed_obs or squared.isna().any():
        return pd.Series(float("nan"), index=returns.index)
    seeded = squared.iloc[seed_obs - 1 :].copy()
    seeded.iloc[0] = squared.iloc[:seed_obs].mean()
    return seeded.ewm(alpha=1 - lam, adjust=False).mean().reindex(returns.index)


def qlike(realised_sq: pd.Series, forecast: pd.Series) -> float:
    """
    Mean QLIKE loss r²/h + ln h (Patton 2011): robust to the noise of r² as a
    proxy for the true variance, and finite when r = 0. Same ranking as the
    r²/h − ln(r²/h) − 1 form. NaN if any forecast is not positive.
    """
    if len(forecast) == 0 or not (forecast > 0).all():
        return float("nan")
    return float((realised_sq / forecast + np.log(forecast)).mean())


def evaluate_variance_forecast(
    returns: pd.Series,
    forecast: pd.Series,
    periods_per_year: float,
    min_obs: int,
) -> ForecastEvaluation:
    """
    Scores ``forecast`` (already aligned: the value at t forecasts ``r_t``)
    on the dates where both are defined and the forecast is positive (a flat
    stretch, e.g. a ledger in cash, forecasts exactly 0 and cannot be scored).
    Mincer-Zarnowitz regression r_t² = a + b·h_t with Newey-West SEs, QLIKE,
    and the RMS of z_t = r_t/√h_t. Below ``min_obs`` such days only ``n`` is set.
    """
    data = pd.concat({"r": returns, "h": forecast}, axis=1).dropna()
    data = data[data["h"] > 0]
    n = len(data)
    if n < min_obs:
        return ForecastEvaluation(n=n)
    r2 = data["r"] ** 2
    scores = dict(
        n=n,
        qlike=qlike(r2, data["h"]),
        # About zero, like the forecast: calibrated when E[z²] = 1. No mean is
        # estimated, so there is no degree of freedom to correct for.
        z_std=float(np.sqrt((data["r"] ** 2 / data["h"]).mean())),
    )
    # A constant forecast leaves the slope undefined (a singular design matrix
    # that inv() does not always reject).
    if data["h"].nunique() < 2:
        return ForecastEvaluation(**scores)
    try:
        fit = hac_ols(r2, data[["h"]])
    except (ValueError, np.linalg.LinAlgError):
        return ForecastEvaluation(**scores)
    return ForecastEvaluation(
        **scores,
        mz_intercept=float(fit.coef[0] * periods_per_year),
        mz_slope=float(fit.coef[1]),
        mz_slope_se=float(fit.se[1]),
        mz_r_squared=float(r2.corr(data["h"]) ** 2),
        hac_lags=fit.lags,
    )
