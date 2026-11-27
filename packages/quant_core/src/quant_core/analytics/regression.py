"""
Linear regression with Newey-West (HAC) standard errors. Shared by the
benchmark statistics (PR 7) and the volatility forecast evaluation (PR 10).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


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
