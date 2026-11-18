"""
Benchmark-relative statistics (PR 7): alignment and capture ratios. The HAC
regression lives in ``regression``.

Align *levels* first, then take returns: each return in both columns then
covers the same interval, and a date missing from either side merges two
intervals instead of inventing a 0% day (which biases beta down).
"""

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
