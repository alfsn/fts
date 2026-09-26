import numpy as np
import pandas as pd
import pytest
from quant_core.calculator import PortfolioCalculator


def test_cagr():
    # 100 to 150 in 2 years
    res = PortfolioCalculator.cagr(100.0, 150.0, 2.0)
    assert np.isclose(res, 0.22474487)  # approx 22.47%


def test_cagr_invalid():
    assert PortfolioCalculator.cagr(-100.0, 150.0, 2.0) == 0.0
    assert PortfolioCalculator.cagr(100.0, 150.0, -1.0) == 0.0


def test_twr():
    returns = [0.10, -0.05, 0.02]
    # (1.10) * (0.95) * (1.02) - 1 = 1.0659 - 1 = 0.0659
    res = PortfolioCalculator.twr(returns)
    assert np.isclose(res, 0.0659)


def test_xirr():
    cash_flows = [
        (pd.Timestamp("2020-01-01"), -1000.0),
        (pd.Timestamp("2021-01-01"), 1100.0),
    ]
    res = PortfolioCalculator.xirr(cash_flows)
    assert np.isclose(res, 0.10, atol=0.01)


def test_xirr_empty():
    assert PortfolioCalculator.xirr([]) == 0.0


def test_sharpe_ratio():
    # Mean return 0.05, std 0.1, risk free 0
    returns = pd.Series([0.05, 0.05, 0.05, -0.05, 0.15])  # mean 0.05, std 0.0707
    res = PortfolioCalculator.sharpe_ratio(
        returns, risk_free_rate=0.0, periods_per_year=12
    )
    # std of sample is ~ 0.07071
    # mean is 0.05
    assert res > 0


def test_sharpe_ratio_empty():
    assert PortfolioCalculator.sharpe_ratio(pd.Series(dtype=float)) == 0.0


def test_max_drawdown():
    # prices: 100 -> 120 -> 90 -> 110
    prices = pd.Series([100, 120, 90, 110])
    # drawdowns:
    # 100/100 - 1 = 0
    # 120/120 - 1 = 0
    # 90/120 - 1 = -0.25
    # 110/120 - 1 = -0.0833
    res = PortfolioCalculator.max_drawdown(prices)
    assert np.isclose(res, -0.25)
