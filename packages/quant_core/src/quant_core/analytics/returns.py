"""
Return primitives (PR 3).

Conventions shared by the whole analytics package:
- Inputs are time-indexed ``pd.Series`` (``DatetimeIndex``); time matters for
  annualisation and cash-flow timing, so bare arrays are not accepted.
- Results are decimal fractions (0.05 == 5%).
- Undefined results are NaN, never 0.0 (0.0 is a legitimate return).
- ``cash_flows`` are external flows from the portfolio's perspective
  (deposit > 0, withdrawal < 0), already reflected in ``values`` at their timestamp.
- A missing valuation (NaN level) is dropped and the next return is chained
  across the gap (V[t+1] / V[t-1] - 1), so the compounded path stays exact.
  Prices are never filled or interpolated. A cash flow needs a valuation at its
  timestamp, so a flow on an unpriced row raises.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize

from .rates import RiskFree, excess_returns

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).total_seconds() / _SECONDS_PER_YEAR


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Observed sampling frequency: ~252 for equity dailies, ~365 for crypto."""
    if len(index) < 2:
        return float("nan")
    years = years_between(index[0], index[-1])
    return (len(index) - 1) / years if years > 0 else float("nan")


def to_daily(series: pd.Series) -> pd.Series:
    """
    Last observation per calendar day. Annualising intraday returns with
    sqrt(N) inflates risk ratios (microstructure noise, autocorrelation), so
    ratios are computed on daily closes.
    """
    return series.groupby(series.index.normalize()).last()


def daily_returns(
    levels: pd.Series, risk_free: RiskFree = 0.0
) -> tuple[pd.Series, float]:
    """
    Returns of the daily closes of ``levels`` (e.g. a flow-adjusted wealth
    index) in excess of ``risk_free`` (0 by default: plain returns), and the
    observed periods per year of those closes. The one path from levels to
    returns for every single-series dispersion statistic (volatility, Sharpe,
    Sortino, EWMA forecast). An interval without a risk-free quote stays NaN.
    """
    daily = to_daily(levels)
    return excess_returns(daily, risk_free), periods_per_year(daily.index)


def _aligned_flows(values: pd.Series, cash_flows: Optional[pd.Series]) -> pd.Series:
    if cash_flows is None:
        return pd.Series(0.0, index=values.index)
    flows = cash_flows.groupby(level=0).sum()
    unmatched = flows.index.difference(values.index)
    if len(unmatched):
        raise ValueError(
            f"Cash flows without a valuation (missing or NaN): {list(unmatched)}"
        )
    return flows.reindex(values.index, fill_value=0.0)


def period_returns(
    values: pd.Series, cash_flows: Optional[pd.Series] = None
) -> pd.Series:
    """Flow-adjusted sub-period returns over the priced rows:
    (V_t - CF_t) / V_{t-1} - 1."""
    values = values.dropna()
    flows = _aligned_flows(values, cash_flows)
    return ((values - flows) / values.shift(1) - 1).iloc[1:]


def wealth_index(
    values: pd.Series, cash_flows: Optional[pd.Series] = None
) -> pd.Series:
    """Growth of 1 unit invested at the first priced row, immune to
    deposits/withdrawals; unpriced rows are left out."""
    values = values.dropna()
    if values.empty:
        return values.astype(float)
    growth = (1 + period_returns(values, cash_flows)).cumprod()
    return pd.concat([pd.Series([1.0], index=values.index[:1]), growth])


def compute_twr(values: pd.Series, cash_flows: Optional[pd.Series] = None) -> float:
    """Cumulative (not annualised) time-weighted return."""
    if values.count() < 2:
        return float("nan")
    return float(wealth_index(values, cash_flows).iloc[-1] - 1)


def compute_cagr(total_return: float, years: float) -> float:
    if not years > 0 or not total_return > -1:
        return float("nan")
    try:
        return float((1 + total_return) ** (1 / years) - 1)
    except OverflowError:  # annualising a gain over minutes
        return float("inf")


def compute_mwr(values: pd.Series, cash_flows: Optional[pd.Series] = None) -> float:
    """
    Annualised money-weighted return (XIRR) over the same inputs as
    ``compute_twr``: the first value is the initial contribution, the last value
    a terminal withdrawal. Flows at the first timestamp are part of that value.
    Unpriced rows are dropped.
    """
    values = values.dropna()
    if len(values) < 2:
        return float("nan")
    investor_flows = -_aligned_flows(values, cash_flows)
    investor_flows.iloc[0] = -values.iloc[0]
    investor_flows.iloc[-1] += values.iloc[-1]
    investor_flows = investor_flows[investor_flows != 0]

    amounts = investor_flows.to_numpy(dtype=float)
    t = np.array([years_between(values.index[0], d) for d in investor_flows.index])

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1 + rate) ** t))

    try:
        # A bracketing solver fails loudly instead of converging to a wrong root.
        return float(optimize.brentq(npv, -0.9999, 1e4))
    except ValueError:
        return float("nan")
