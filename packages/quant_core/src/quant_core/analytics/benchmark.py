"""
Benchmark-relative statistics (PR 7): alignment, OLS with Newey-West (HAC)
standard errors, and capture ratios.

Align *levels* first, then take returns: each return in both columns then
covers the same interval, and a date missing from either side merges two
intervals instead of inventing a 0% day (which biases beta down).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .returns import to_daily


def align_daily(portfolio: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Daily closes of both series on the dates where both have a real value."""
    return pd.concat(
        {
            "portfolio": to_daily(portfolio.dropna()),
            "benchmark": to_daily(benchmark.dropna()),
        },
        axis=1,
        join="inner",
    )


def newey_west_lags(n: int) -> int:
    """Newey-West (1994) automatic bandwidth: floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


@dataclass(frozen=True)
class HacFit:
    """OLS of y on [1, X]: ``coef[0]`` is the intercept, ``se`` is HAC."""

    coef: np.ndarray
    se: np.ndarray
    n: int
    lags: int
    df: int
    fitted: pd.Series


def hac_ols(y: pd.Series, X: pd.DataFrame, lags: Optional[int] = None) -> HacFit:
    """
    OLS with an intercept and Newey-West standard errors (Bartlett kernel,
    small-sample factor n / (n - k)): robust to the autocorrelation and
    heteroskedasticity of daily returns. Rows with any NaN are dropped.
    Matches statsmodels ``OLS(...).fit(cov_type="HAC",
    cov_kwds={"maxlags": L, "use_correction": True})``.
    """
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    n, k = len(data), X.shape[1] + 1
    if n <= k:
        raise ValueError(f"{n} observations for {k} coefficients.")
    yv = data["y"].to_numpy(dtype=float)
    Xv = np.column_stack([np.ones(n), data.drop(columns="y").to_numpy(dtype=float)])

    bread = np.linalg.inv(Xv.T @ Xv)
    coef = bread @ Xv.T @ yv
    scores = Xv * (yv - Xv @ coef)[:, None]
    lags = newey_west_lags(n) if lags is None else lags
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    cov = bread @ meat @ bread * n / (n - k)
    return HacFit(
        coef=coef,
        se=np.sqrt(np.diag(cov)),
        n=n,
        lags=lags,
        df=n - k,
        fitted=pd.Series(Xv @ coef, index=data.index),
    )


def _geometric_mean(returns: pd.Series) -> float:
    return float(np.expm1(np.log1p(returns).mean()))


def _capture(returns: pd.Series, benchmark: pd.Series) -> float:
    if len(benchmark) == 0:
        return float("nan")
    denominator = _geometric_mean(benchmark)
    if not abs(denominator) > 0:
        return float("nan")
    return _geometric_mean(returns) / denominator


def compute_capture_ratios(
    returns: pd.Series, benchmark_returns: pd.Series
) -> tuple[float, float]:
    """
    (up, down) capture: the portfolio's geometric mean return over the
    benchmark's up (> 0) or down (< 0) periods, divided by the benchmark's
    (Morningstar's form). Pass total returns, not excess: subtracting a rate
    would move periods between up and down. Zero periods count as neither.
    """
    r, b = returns.align(benchmark_returns, join="inner")
    keep = r.notna() & b.notna()
    r, b = r[keep], b[keep]
    return _capture(r[b > 0], b[b > 0]), _capture(r[b < 0], b[b < 0])
