import numpy as np
import pandas as pd
import pytest
from quant_core.analytics import (
    compute_cagr,
    compute_calmar,
    compute_information_ratio,
    compute_max_drawdown,
    compute_mwr,
    compute_rolling_vol,
    compute_sharpe,
    compute_sortino,
    compute_twr,
    compute_win_rate,
    drawdown_episodes,
    max_time_underwater_days,
    periods_per_year,
    round_trip_pnls,
    to_daily,
    top_drawdowns,
)


def _series(values, start="2024-01-01", freq="D"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float
    )


# --- Returns -----------------------------------------------------------------


def test_twr_without_flows_is_simple_return():
    assert compute_twr(_series([100, 110, 104.5, 106.59])) == pytest.approx(0.0659)


def test_twr_neutralises_deposits():
    # 100 -> 110 (+10%), then deposit 100 -> 210, then 210 -> 231 (+10%)
    values = _series([100, 110, 210, 231])
    flows = pd.Series([100.0], index=[values.index[2]])
    assert compute_twr(values, flows) == pytest.approx(1.1 * 1.1 - 1)
    # Naive first/last would report 131% "return" from a deposit.
    assert compute_twr(values) > 1.0


def test_twr_rejects_flows_without_valuation():
    values = _series([100, 110])
    with pytest.raises(ValueError):
        compute_twr(values, pd.Series([5.0], index=[pd.Timestamp("2030-01-01")]))


def test_twr_undefined_for_single_point():
    assert np.isnan(compute_twr(_series([100])))


def test_mwr_single_period_matches_annual_return():
    values = pd.Series(
        [1000.0, 1100.0],
        index=pd.to_datetime(["2020-01-01", "2021-01-01"]),
    )
    assert compute_mwr(values) == pytest.approx(0.10, abs=1e-3)


def test_mwr_penalises_badly_timed_deposit():
    # Deposit right before a loss: MWR < TWR.
    values = pd.Series(
        [100.0, 120.0, 220.0, 176.0],
        index=pd.to_datetime(
            [
                "2020-01-01 00:00",
                "2020-07-01 00:00",
                "2020-07-01 01:00",
                "2021-01-01 00:00",
            ]
        ),
    )
    flows = pd.Series([100.0], index=[values.index[2]])
    twr = compute_twr(values, flows)
    assert twr == pytest.approx(1.2 * 0.8 - 1)
    assert compute_mwr(values, flows) < twr


def test_mwr_is_nan_when_unsolvable():
    assert np.isnan(compute_mwr(_series([100])))


def test_cagr():
    assert compute_cagr(0.5, 2.0) == pytest.approx(0.22474487)
    assert np.isnan(compute_cagr(0.5, 0.0))
    assert np.isnan(compute_cagr(-1.0, 1.0))


def test_periods_per_year_infers_calendar():
    assert periods_per_year(pd.date_range("2024-01-01", periods=366, freq="D")) == (
        pytest.approx(365.25, rel=0.01)
    )
    business = pd.bdate_range("2024-01-01", "2024-12-31")
    assert periods_per_year(business) == pytest.approx(262, rel=0.02)


def test_to_daily_keeps_last_tick_per_day():
    s = _series([1, 2, 3, 4], start="2024-01-01 22:00", freq="h")
    assert list(to_daily(s)) == [2.0, 4.0]


# --- Drawdowns ---------------------------------------------------------------


def test_max_drawdown():
    assert compute_max_drawdown(_series([100, 120, 90, 110])) == pytest.approx(-0.25)


def test_drawdown_episodes_and_recovery():
    values = _series([100, 90, 100, 120, 60, 80, 130, 125])
    episodes = drawdown_episodes(values)
    assert [e.depth for e in episodes] == pytest.approx([-0.1, -0.5, 125 / 130 - 1])
    first, second, last = episodes
    assert (first.peak, first.trough, first.recovery) == tuple(values.index[[0, 1, 2]])
    assert second.duration_days == 3
    assert last.recovery is None and last.duration_days == 1


def test_top_drawdowns_sorted_by_depth():
    values = _series([100, 90, 100, 120, 60, 80, 130, 125])
    assert [e.depth for e in top_drawdowns(values, n=2)] == pytest.approx([-0.5, -0.1])


def test_time_underwater_counts_unrecovered_tail():
    assert max_time_underwater_days(_series([100, 90, 95, 99, 98])) == 4
    assert max_time_underwater_days(_series([1, 2, 3])) == 0


# --- Ratios ------------------------------------------------------------------


def test_sharpe_uses_sample_std_and_annualises():
    r = _series([0.01, -0.005, 0.02, 0.0])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert compute_sharpe(r, 252) == pytest.approx(expected)


def test_sharpe_undefined_cases_are_nan():
    assert np.isnan(compute_sharpe(_series([]), 252))
    assert np.isnan(compute_sharpe(_series([0.01, 0.01]), 252))


def test_sortino_only_penalises_downside():
    r = _series([0.02, -0.01, 0.03, -0.02])
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert compute_sortino(r, 252) == pytest.approx(r.mean() / downside * np.sqrt(252))
    assert np.isnan(compute_sortino(_series([0.01, 0.02]), 252))


def test_calmar():
    assert compute_calmar(0.2, -0.1) == pytest.approx(2.0)
    assert np.isnan(compute_calmar(0.2, 0.0))


def test_information_ratio_aligns_by_date():
    r = _series([0.01, 0.02, 0.03])
    b = _series([0.0, 0.01, 0.01, 0.5], start="2023-12-31")  # one day earlier
    active = (r - b).dropna()
    expected = active.mean() / active.std(ddof=1) * np.sqrt(252)
    assert compute_information_ratio(r, b, 252) == pytest.approx(expected)


def test_rolling_vol_window():
    r = _series(np.random.default_rng(0).normal(0, 0.01, 40))
    vol = compute_rolling_vol(r, 30, 252)
    assert vol.iloc[:29].isna().all()
    assert vol.iloc[-1] == pytest.approx(r.iloc[-30:].std(ddof=1) * np.sqrt(252))


# --- Trades ------------------------------------------------------------------


def test_round_trips_fifo_long_and_short():
    fills = [(1, 10), (1, 12), (-2, 11), (-1, 20), (1, 15)]
    # close 1@10 and 1@12 at 11 -> +1 - 1 = 0; short 1@20 covered at 15 -> +5
    assert round_trip_pnls(fills) == pytest.approx([0.0, 5.0])


def test_open_fills_are_not_wins():
    assert round_trip_pnls([(1, 10), (2, 11)]) == []
    assert np.isnan(compute_win_rate([]))
    assert compute_win_rate([1.0, -1.0, 2.0]) == pytest.approx(2 / 3)


def test_daily_returns_are_the_daily_closes_returns_and_their_frequency():
    from quant_core.analytics import daily_returns, periods_per_year, to_daily

    idx = pd.date_range("2024-01-01", periods=96, freq="6h")
    levels = pd.Series(np.linspace(1.0, 1.2, 96), index=idx)
    returns, ppy = daily_returns(levels)
    daily = to_daily(levels)
    pd.testing.assert_series_equal(returns, daily.pct_change().dropna())
    assert ppy == periods_per_year(daily.index)


def test_daily_returns_in_excess_of_a_rate_keep_a_stale_rate_as_nan():
    from quant_core.analytics import daily_returns, excess_returns, to_daily

    # the second interval starts on Jan 15, 13 days after the only quote
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-15", "2024-01-16"], tz="UTC")
    levels = pd.Series([1.0, 1.01, 1.02], idx)
    rates = pd.Series([0.04], pd.DatetimeIndex(["2024-01-02"], tz="UTC"))
    excess, _ = daily_returns(levels, rates)
    pd.testing.assert_series_equal(excess, excess_returns(to_daily(levels), rates))
    assert len(excess) == 2 and np.isnan(excess.iloc[1])  # kept, not dropped
