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


def test_dispersion_at_zero_rate_is_plain_daily_returns():
    from quant_core.analytics import periods_per_year

    idx = pd.date_range("2024-01-01", periods=200, freq="6h")
    values = pd.Series(
        100 * np.cumprod(1 + np.random.default_rng(7).normal(0, 0.01, 200)), idx
    )
    m = compute_performance_metrics(values)
    daily = values.groupby(values.index.normalize()).last()
    r = daily.pct_change().dropna()
    ppy = periods_per_year(daily.index)
    assert m.volatility == pytest.approx(r.std(ddof=1) * np.sqrt(ppy))
    assert m.sharpe_ratio == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


# --- unpriced valuations: dropped and chained across, never filled ----------


def _random_walk(n: int = 400) -> pd.Series:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_an_unpriced_day_is_chained_across_not_lost():
    values = _random_walk()
    holed = values.copy()
    holed.iloc[200] = np.nan
    m = compute_performance_metrics(holed)
    dropped = compute_performance_metrics(values.drop(values.index[200]))
    # Same as deleting the row: the move across the gap is kept.
    assert m.total_return == pytest.approx(values.iloc[-1] / values.iloc[0] - 1)
    assert m.max_drawdown == pytest.approx(dropped.max_drawdown)
    assert m.sharpe_ratio == pytest.approx(dropped.sharpe_ratio)
    assert np.isfinite(m.sortino_ratio)
    assert m.unpriced_obs == 1 and dropped.unpriced_obs == 0


def test_unpriced_first_and_last_rows_are_skipped():
    values = _random_walk(50)
    holed = values.copy()
    holed.iloc[[0, -1]] = np.nan
    m = compute_performance_metrics(holed)
    assert m.total_return == pytest.approx(values.iloc[-2] / values.iloc[1] - 1)
    assert m.unpriced_obs == 2


def test_all_unpriced_is_nan_with_the_count():
    values = pd.Series(np.nan, index=pd.bdate_range("2024-01-01", periods=3))
    m = compute_performance_metrics(values)
    assert np.isnan(m.total_return) and m.unpriced_obs == 3


def test_a_cash_flow_on_an_unpriced_day_raises():
    values = _random_walk(10)
    values.iloc[5] = np.nan
    flows = pd.Series([50.0], index=[values.index[5]])
    with pytest.raises(ValueError, match="without a valuation"):
        compute_performance_metrics(values, flows)


def test_a_missing_rate_still_makes_sharpe_undefined():
    values = _random_walk(30)
    rates = pd.Series(0.02, index=values.index[:10])  # quotes stop on day 10
    m = compute_performance_metrics(values, risk_free_rate=rates)
    assert np.isnan(m.sharpe_ratio) and np.isfinite(m.volatility)
