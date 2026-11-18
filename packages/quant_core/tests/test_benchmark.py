"""PR 7: aligned benchmark, HAC alpha/beta, Dimson beta, captures, IR, risk-free."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from quant_core.analytics import (
    align_daily,
    compute_capture_ratios,
    compute_sharpe,
    excess_returns,
    hac_ols,
    net_of_withholding,
    newey_west_lags,
    tbill_bond_equivalent_yield,
)
from quant_core.enums import CorporateActionType
from quant_core.metrics import compute_benchmark_metrics, compute_performance_metrics
from quant_core.models import BenchmarkMetrics, CorporateAction


def _market(n=600, beta=0.8, alpha_daily=0.0003, lag_beta=0.0, seed=3):
    """Benchmark and portfolio price levels with a known daily alpha and beta."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n + 1, tz="UTC")
    b = rng.normal(0.0004, 0.01, n)
    lagged = np.concatenate([[0.0], b[:-1]])
    p = alpha_daily + beta * b + lag_beta * lagged + rng.normal(0, 0.002, n)

    def level(r):
        return pd.Series(100 * np.cumprod(np.r_[1.0, 1 + r]), index=dates)

    return level(p), level(b)


# --- HAC OLS ---------------------------------------------------------------------


def test_hac_ols_matches_statsmodels():
    """Reference numbers from statsmodels 0.14 OLS(...).fit(cov_type="HAC",
    cov_kwds={"maxlags": 5, "use_correction": True}) on the same data."""
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0, 0.01, n)
    e = rng.normal(0, 0.005, n)
    y = 0.0002 + 1.3 * x + e + 0.5 * np.concatenate([[0], e[:-1]])  # MA(1) errors
    idx = pd.bdate_range("2022-01-03", periods=n)

    fit = hac_ols(pd.Series(y, idx), pd.DataFrame({"b": x}, idx))

    assert fit.lags == newey_west_lags(n) == 5
    assert fit.n == n and fit.df == n - 2
    np.testing.assert_allclose(
        fit.coef, [-0.00082488098765032725, 1.2876619851985356], rtol=1e-10
    )
    np.testing.assert_allclose(
        fit.se, [0.00035550389143120198, 0.032903979577282465], rtol=1e-10
    )


def test_hac_ols_refuses_too_few_observations():
    idx = pd.bdate_range("2024-01-01", periods=2)
    with pytest.raises(ValueError, match="2 observations"):
        hac_ols(pd.Series([0.1, 0.2], idx), pd.DataFrame({"b": [0.1, 0.3]}, idx))


# --- alignment ---------------------------------------------------------------------


def test_align_daily_drops_dates_missing_on_either_side():
    """A US holiday is dropped, never a fake 0% benchmark day."""
    dates = pd.bdate_range("2024-07-01", periods=5, tz="UTC")  # Jul 4 is dates[3]
    portfolio = pd.Series([100.0, 101, 102, 103, 104], index=dates)
    spy = pd.Series([50.0, 51, 52, np.nan, 54], index=dates)

    aligned = align_daily(portfolio, spy)

    assert list(aligned.index) == [dates[0], dates[1], dates[2], dates[4]]
    returns = aligned.pct_change().iloc[1:]
    assert returns["benchmark"].iloc[-1] == pytest.approx(54 / 52 - 1)
    assert returns["portfolio"].iloc[-1] == pytest.approx(104 / 102 - 1)
    assert not (returns["benchmark"] == 0).any()


def test_a_deposit_on_a_dropped_date_is_not_a_return():
    """Wealth is flow-adjusted before aligning, so the flow is never lost."""
    dates = pd.bdate_range("2024-07-01", periods=4, tz="UTC")
    values = pd.Series([100.0, 100.0, 1100.0, 1100.0], index=dates)
    flows = pd.Series([1000.0], index=[dates[2]])
    spy = pd.Series([50.0, 50.0, np.nan, 50.0], index=dates)

    analysis = compute_benchmark_metrics(values, flows, spy, min_obs=1)

    assert list(analysis.wealth["portfolio"]) == pytest.approx([1.0, 1.0, 1.0])


# --- alpha / beta ------------------------------------------------------------------


def test_recovers_known_alpha_and_beta():
    portfolio, spy = _market()
    m = compute_benchmark_metrics(portfolio, None, spy).metrics

    ppy = 600 / ((spy.index[-1] - spy.index[0]).total_seconds() / (365.25 * 86400))
    assert m.n_obs == 600
    assert m.beta == pytest.approx(0.8, abs=0.02)
    assert m.beta_ci_low < 0.8 < m.beta_ci_high
    assert m.alpha == pytest.approx(0.0003 * ppy, abs=3 * m.alpha_se)
    assert m.alpha_ci_low < m.alpha < m.alpha_ci_high
    assert m.hac_lags == newey_west_lags(600)
    assert 0.9 < m.r_squared < 1


def test_dimson_beta_picks_up_a_lagged_response():
    """Stale prices: half the reaction lands a day late. Plain beta misses it."""
    portfolio, spy = _market(beta=0.5, lag_beta=0.5, alpha_daily=0.0)
    m = compute_benchmark_metrics(portfolio, None, spy).metrics

    assert m.beta == pytest.approx(0.5, abs=0.03)
    assert m.dimson_beta == pytest.approx(1.0, abs=0.05)


def test_below_min_obs_everything_is_nan_but_n():
    portfolio, spy = _market(n=100)
    analysis = compute_benchmark_metrics(portfolio, None, spy)
    m = analysis.metrics

    assert m.n_obs == 100
    assert m.hac_lags is None
    for name in ("alpha", "beta", "dimson_beta", "up_capture", "information_ratio"):
        assert np.isnan(getattr(m, name)), name
    assert analysis.excess["fitted"].isna().all()
    assert len(analysis.wealth) == 101  # the chart still draws


def test_missing_risk_free_leaves_alpha_beta_and_sharpe_undefined():
    portfolio, spy = _market()
    m = compute_benchmark_metrics(portfolio, None, spy, float("nan")).metrics

    assert np.isnan(m.alpha) and np.isnan(m.beta) and np.isnan(m.dimson_beta)
    assert np.isfinite(m.information_ratio)
    assert np.isfinite(m.up_capture) and np.isfinite(m.r_squared)
    assert np.isnan(
        compute_performance_metrics(portfolio, None, float("nan")).sharpe_ratio
    )


def test_beta_is_unaffected_by_a_constant_risk_free_rate_on_equal_intervals():
    portfolio, spy = _market()
    zero = compute_benchmark_metrics(portfolio, None, spy).metrics
    four = compute_benchmark_metrics(portfolio, None, spy, 0.04).metrics
    assert four.beta == pytest.approx(zero.beta, rel=1e-3)
    # Jensen: alpha = rp - beta rb - rf (1 - beta), so it falls by about rf (1 - beta)
    assert zero.alpha - four.alpha == pytest.approx(0.04 * (1 - zero.beta), abs=0.005)


def test_benchmark_metrics_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        BenchmarkMetrics(alpah=0.1)


# --- capture ratios / IR -----------------------------------------------------------


def test_capture_ratios_by_hand():
    idx = pd.bdate_range("2024-01-01", periods=4)
    bench = pd.Series([0.02, -0.01, 0.0, 0.04], idx)
    port = pd.Series([0.01, -0.02, 0.05, 0.03], idx)

    up, down = compute_capture_ratios(port, bench)

    def geo(r):
        return np.prod(1 + np.array(r)) ** (1 / len(r)) - 1

    assert up == pytest.approx(geo([0.01, 0.03]) / geo([0.02, 0.04]))
    assert down == pytest.approx(-0.02 / -0.01)  # zero day counts as neither


def test_capture_ratio_without_down_days_is_nan():
    idx = pd.bdate_range("2024-01-01", periods=2)
    up, down = compute_capture_ratios(
        pd.Series([0.01, 0.02], idx), pd.Series([0.01, 0.01], idx)
    )
    assert up == pytest.approx(1.5, rel=0.01) and np.isnan(down)


# --- risk-free -------------------------------------------------------------------


def test_tbill_bond_equivalent_yield():
    assert tbill_bond_equivalent_yield(0.0412) == pytest.approx(
        365 * 0.0412 / (360 - 91 * 0.0412)
    )
    assert tbill_bond_equivalent_yield(0.0412) == pytest.approx(0.04222, abs=1e-5)


def test_risk_free_accrues_over_actual_days_from_the_rate_at_the_start():
    # Fri -> Mon is 3 days; the Monday quote (a jump) must not be used for it.
    idx = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08"], tz="UTC")
    levels = pd.Series([100.0, 100.0, 100.0], idx)
    rates = pd.Series([0.0365, 0.073, 0.5], idx)

    excess = excess_returns(levels, rates)

    assert list(excess) == pytest.approx([-0.0365 / 365, -0.073 * 3 / 365])


def test_a_stale_rate_is_nan():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-20"], tz="UTC")
    rates = pd.Series([0.04], pd.DatetimeIndex(["2024-01-02"], tz="UTC"))
    excess = excess_returns(pd.Series([1.0, 1.0, 1.0], idx), rates)
    assert excess.iloc[0] == pytest.approx(-0.04 / 365)
    # the Jan 3 quote is fresh (carried 1 day), the interval is 17 days long
    assert excess.iloc[1] == pytest.approx(-0.04 * 17 / 365)
    portfolio_after_gap = excess_returns(
        pd.Series([1.0, 1.0], pd.DatetimeIndex(["2024-01-20", "2024-01-21"], tz="UTC")),
        rates,
    )
    assert np.isnan(portfolio_after_gap.iloc[0])


def test_sharpe_with_a_constant_rate_series_equals_the_float():
    portfolio, _ = _market()
    rates = pd.Series(0.04, index=portfolio.index)
    as_float = excess_returns(portfolio, 0.04)
    as_series = excess_returns(portfolio, rates)
    assert compute_sharpe(as_series, 252) == pytest.approx(
        compute_sharpe(as_float, 252)
    )


# --- withholding -----------------------------------------------------------------


def test_net_of_withholding_scales_dividends_only():
    from datetime import datetime, timezone

    ex = datetime(2024, 3, 15, tzinfo=timezone.utc)
    actions = [
        CorporateAction(
            instrument_id="SPY",
            action_type=CorporateActionType.DIVIDEND,
            ex_date=ex,
            amount=2.0,
        ),
        CorporateAction(
            instrument_id="SPY",
            action_type=CorporateActionType.SPLIT,
            ex_date=ex,
            ratio=2.0,
        ),
    ]
    net = net_of_withholding(actions, 0.30)
    assert net[0].amount == pytest.approx(1.4)
    assert net[1] == actions[1]
    assert actions[0].amount == 2.0  # input untouched
    with pytest.raises(ValueError):
        net_of_withholding(actions, 1.0)
