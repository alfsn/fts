"""
Risk and risk-adjusted ratios (PR 4 & PR 6).

``periods_per_year`` is required on purpose: a silent 252 default is wrong for
crypto (365) and for anything that is not daily. Use ``returns.periods_per_year``.
Sample statistics use ddof=1.
"""

import numpy as np
import pandas as pd


def compute_volatility(returns: pd.Series, periods_per_year: float) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def compute_rolling_vol(
    returns: pd.Series, window: int, periods_per_year: float
) -> pd.Series:
    """Annualised rolling volatility; NaN until ``window`` observations exist."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def _annualised_ratio(numerator: pd.Series, denominator: float, ppy: float) -> float:
    if len(numerator) < 2 or not denominator > 0:
        return float("nan")
    return float(numerator.mean() / denominator * np.sqrt(ppy))


def compute_sharpe(
    returns: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0
) -> float:
    """``risk_free_rate`` is annual; it is de-annualised per period."""
    excess = returns - risk_free_rate / periods_per_year
    return _annualised_ratio(excess, excess.std(ddof=1), periods_per_year)


def compute_sortino(
    returns: pd.Series, periods_per_year: float, target_return: float = 0.0
) -> float:
    """Downside deviation is taken over *all* periods (not only losing ones)."""
    excess = returns - target_return / periods_per_year
    downside_dev = np.sqrt((excess.clip(upper=0) ** 2).mean())
    return _annualised_ratio(excess, downside_dev, periods_per_year)


def compute_calmar(cagr: float, max_drawdown: float) -> float:
    if not max_drawdown < 0:
        return float("nan")
    return float(cagr / abs(max_drawdown))


def compute_information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: float
) -> float:
    """Aligns on timestamps; positional subtraction would mix different dates."""
    active = returns.sub(benchmark_returns).dropna()
    return _annualised_ratio(active, active.std(ddof=1), periods_per_year)
