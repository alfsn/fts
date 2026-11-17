"""
Risk-free rates and excess returns (PR 7).

Rates are annual decimal fractions (0.04 == 4%). They accrue simple interest
ACT/365 over each interval's actual length, not 1/periods_per_year: aligned
daily series have unequal intervals (weekends, holidays, dropped days).
"""

from typing import Union

import pandas as pd

RiskFree = Union[float, pd.Series]

TBILL_DAYS = 91  # 13-week bill
RATE_STALENESS = pd.Timedelta(days=7)  # an older quote is NaN, not carried


def tbill_bond_equivalent_yield(discount: RiskFree, days: int = TBILL_DAYS) -> RiskFree:
    """
    Bond-equivalent (ACT/365) yield of a T-bill quoted as a bank discount
    yield (ACT/360 on face value), e.g. Yahoo's ``^IRX`` / 100:
    ``365 d / (360 - days d)``. 4.12% discount -> 4.22% BEY.
    """
    return 365 * discount / (360 - days * discount)


def excess_returns(levels: pd.Series, risk_free: RiskFree = 0.0) -> pd.Series:
    """
    Period returns of ``levels`` minus the risk-free return over the same
    interval. A rate Series is taken as of each interval's *start* (the last
    quote at most ``RATE_STALENESS`` old), so there is no look-ahead; an
    interval without a rate is NaN.
    """
    returns = levels.pct_change().iloc[1:]
    days = levels.index.to_series().diff().dt.total_seconds().iloc[1:] / 86400
    if isinstance(risk_free, pd.Series):
        rate = (
            risk_free.dropna()
            .sort_index()
            .reindex(levels.index, method="ffill", tolerance=RATE_STALENESS)
        )
        rate = rate.shift(1).iloc[1:]
    else:
        rate = risk_free
    return returns - rate * days / 365
