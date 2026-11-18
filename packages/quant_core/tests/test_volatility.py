"""PR 10: EWMA volatility forecast and its out-of-sample evaluation."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from quant_core.analytics import (
    RISKMETRICS_DAILY_LAMBDA,
    annualise_variance,
    evaluate_variance_forecast,
    ewma_variance,
    qlike,
)
from quant_core.metrics import MIN_VOL_EVAL_OBS, compute_vol_forecast
from quant_core.models import ForecastEvaluation, VolForecastMetrics


def _returns(n, seed=0, vol=0.01, freq="B"):
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    return pd.Series(np.random.default_rng(seed).normal(0, vol, n), index=idx)


def _garch_values(n=1500, seed=0, freq="B"):
    """Prices driven by GARCH(1,1) returns: volatility clusters, so a good
    forecaster beats a flat one."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    h, r = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * r[t] ** 2 + beta * h
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    return pd.Series(100 * np.cumprod(1 + r), index=idx)


# --- ewma_variance -----------------------------------------------------------


def test_ewma_matches_the_recursion_and_its_closed_form():
    r = _returns(80)
    lam, seed = 0.94, 30
    s = ewma_variance(r, lam, seed)

    expected = [(r.iloc[:seed] ** 2).mean()]
    for x in r.iloc[seed:]:
        expected.append(lam * expected[-1] + (1 - lam) * x**2)
    np.testing.assert_allclose(s.iloc[seed - 1 :], expected, rtol=1e-12)

    # closed form: lam^k · seed + (1 - lam) Σ lam^j r²_{T-j}
    k = len(r) - seed
    tail = r.iloc[seed:].to_numpy() ** 2
    weights = (1 - lam) * lam ** np.arange(k)[::-1]
    closed = lam**k * expected[0] + (weights * tail).sum()
    assert s.iloc[-1] == pytest.approx(closed, rel=1e-12)


def test_ewma_is_nan_before_the_seed_and_on_short_history():
    r = _returns(40)
    s = ewma_variance(r, seed_obs=30)
    assert s.iloc[:29].isna().all() and s.iloc[29:].notna().all()
    assert ewma_variance(r.iloc[:29], seed_obs=30).isna().all()


def test_ewma_has_no_look_ahead():
    r = _returns(60)
    shocked = r.copy()
    shocked.iloc[45] = 0.2
    before, after = ewma_variance(r).shift(1), ewma_variance(shocked).shift(1)
    # the forecast *for* day 45 was made at day 44: the shock can't move it
    assert after.iloc[45] == before.iloc[45]
    assert after.iloc[46] > before.iloc[46]


@pytest.mark.parametrize("lam", [0.0, 1.0, -0.1, 1.5])
def test_ewma_rejects_a_decay_outside_zero_one(lam):
    with pytest.raises(ValueError):
        ewma_variance(_returns(40), lam)


def test_default_decay_is_riskmetrics_daily():
    assert RISKMETRICS_DAILY_LAMBDA == 0.94


def test_annualise_variance():
    assert annualise_variance(0.0001, 252) == pytest.approx(0.01 * np.sqrt(252))


# --- qlike / evaluate --------------------------------------------------------


def test_qlike_is_finite_on_a_zero_return_and_nan_on_a_zero_forecast():
    h = pd.Series([1e-4, 1e-4])
    assert np.isfinite(qlike(pd.Series([0.0, 1e-4]), h))
    assert np.isnan(qlike(pd.Series([1e-4, 1e-4]), pd.Series([0.0, 1e-4])))


def test_qlike_is_minimised_by_the_true_variance():
    r2 = _returns(5000, vol=0.01) ** 2
    losses = {v: qlike(r2, pd.Series(v, index=r2.index)) for v in (5e-5, 1e-4, 2e-4)}
    assert min(losses, key=losses.get) == 1e-4


def test_evaluation_below_the_minimum_is_nan_with_n_set():
    r = _returns(100)
    e = evaluate_variance_forecast(r, ewma_variance(r).shift(1), 252, min_obs=252)
    assert e.n == 70
    assert np.isnan(e.qlike) and np.isnan(e.mz_slope) and np.isnan(e.z_std)


def test_a_constant_forecast_has_no_slope_but_a_loss():
    r = _returns(400)
    e = evaluate_variance_forecast(r, pd.Series(1e-4, index=r.index), 252, 252)
    assert np.isfinite(e.qlike) and np.isfinite(e.z_std)
    assert np.isnan(e.mz_slope) and e.hac_lags is None


# --- compute_vol_forecast ----------------------------------------------------


def test_ewma_beats_the_rolling_baseline_on_clustered_volatility():
    a = compute_vol_forecast(_garch_values())
    m = a.metrics
    assert m.ewma.n == m.baseline.n >= MIN_VOL_EVAL_OBS  # same days for both
    assert m.ewma.qlike < m.baseline.qlike
    # informative, though below 1: λ = 0.94 over-reacts relative to this GARCH
    assert m.ewma.mz_slope > 3 * m.ewma.mz_slope_se
    assert 0.9 < m.ewma.z_std < 1.1
    assert m.lam == 0.94 and m.forecast_vol > 0
    assert list(a.frame.columns) == ["forecast", "realised", "baseline"]
    # the realised line covers the *next* horizon, so it ends early
    assert a.frame["realised"].iloc[-m.horizon :].isna().all()


def test_windows_follow_the_observed_calendar():
    equity = compute_vol_forecast(_garch_values(800, freq="B")).metrics
    crypto = compute_vol_forecast(_garch_values(800, freq="D")).metrics
    assert 250 <= equity.baseline_window <= 262 and 20 <= equity.horizon <= 22
    assert (crypto.baseline_window, crypto.horizon) == (365, 30)


def test_crypto_forecast_is_annualised_with_365():
    values = _garch_values(800, freq="D")
    a = compute_vol_forecast(values)
    r = values.pct_change().dropna()
    daily_var = ewma_variance(r).iloc[-1]
    assert a.metrics.forecast_vol == pytest.approx(np.sqrt(daily_var * 365), rel=1e-3)


def test_a_deposit_is_not_volatility():
    values = _garch_values(400)
    flows = pd.Series([50.0], index=[values.index[200]])
    with_deposit = values.copy()
    with_deposit.iloc[200:] *= (values.iloc[200] + 50) / values.iloc[200]
    clean = compute_vol_forecast(values)
    funded = compute_vol_forecast(with_deposit, flows)
    pd.testing.assert_frame_equal(clean.frame, funded.frame, rtol=1e-9)


def test_intraday_values_give_the_daily_forecast():
    daily = _garch_values(400)
    noon = daily.copy()
    noon.index = noon.index + pd.Timedelta(hours=12)
    intraday = pd.concat([daily * 0.97, noon]).sort_index()  # close = noon value
    a, b = compute_vol_forecast(noon), compute_vol_forecast(intraday)
    assert a.metrics.forecast_vol == pytest.approx(b.metrics.forecast_vol)


def test_short_history_reports_counts_and_nan():
    m = compute_vol_forecast(_garch_values(200)).metrics
    assert m.n_obs == 199 and m.ewma.n == m.baseline.n == 0
    assert np.isnan(m.ewma.qlike) and np.isnan(m.baseline.qlike)
    assert m.forecast_vol > 0  # the forecast itself only needs the seed
    assert compute_vol_forecast(_garch_values(1)).metrics.n_obs == 0


def test_models_reject_unknown_fields():
    with pytest.raises(ValidationError):
        VolForecastMetrics(volatility=0.1)
    with pytest.raises(ValidationError):
        ForecastEvaluation(slope=1.0)
