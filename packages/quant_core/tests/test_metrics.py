import numpy as np
import pandas as pd
import pytest
from quant_core.metrics import compute_performance_metrics


def test_metrics_daily_series():
    values = pd.Series(
        [100.0, 110.0, 99.0, 120.0],
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    m = compute_performance_metrics(values)
    assert m.total_return == pytest.approx(0.20)
    assert m.max_drawdown == pytest.approx(-0.10)
    assert m.time_underwater_days == pytest.approx(2.0)
    assert m.sharpe_ratio > 0
    assert m.calmar_ratio == pytest.approx(m.cagr / 0.10)


def test_metrics_intraday_backtest_has_no_annualised_dispersion():
    # A same-day backtest yields one daily close: Sharpe is undefined, not 0.
    values = pd.Series(
        [100.0, 110.0, 105.0, 120.0],
        index=pd.date_range("2024-01-01 10:00", periods=4, freq="min"),
    )
    m = compute_performance_metrics(values)
    assert m.total_return == pytest.approx(0.20)
    assert m.max_drawdown == pytest.approx(105 / 110 - 1)
    assert np.isnan(m.sharpe_ratio)


def test_metrics_cash_flows_do_not_count_as_performance():
    values = pd.Series(
        [100.0, 100.0, 200.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    flows = pd.Series([100.0], index=[values.index[2]])
    assert compute_performance_metrics(values, flows).total_return == pytest.approx(0.0)


def test_metrics_empty():
    m = compute_performance_metrics(pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    assert np.isnan(m.total_return)
    assert m.total_trades == 0
