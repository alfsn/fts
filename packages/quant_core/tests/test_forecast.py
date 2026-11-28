import numpy as np
import pandas as pd
import pytest
from quant_core.analytics import hit_rate, ic_se_newey_west, rank_ic
from quant_core.metrics import compute_fold_ic_summary, compute_forecast_metrics
from scipy.stats import spearmanr


def _fixture():
    """Signal plus MA(2) noise, as in the statsmodels reference run."""
    rng = np.random.default_rng(11)
    n = 120
    f = rng.normal(size=n)
    e = rng.normal(size=n)
    noise = e + 0.6 * np.r_[0, e[:-1]] + 0.3 * np.r_[0, 0, e[:-2]]
    return f, 0.3 * f + noise


def test_rank_ic_matches_spearmanr_with_ties_and_nan_pairs():
    pred = np.array([0.1, 0.1, 0.3, np.nan, 0.2, 0.5, 0.5, -0.1])
    target = np.array([0.02, -0.01, 0.03, 0.04, np.nan, 0.01, 0.05, -0.02])
    keep = ~(np.isnan(pred) | np.isnan(target))

    expected = spearmanr(pred[keep], target[keep]).statistic
    assert rank_ic(pred, target) == pytest.approx(expected, rel=1e-12)


def test_constant_predictions_are_nan_not_zero():
    pred = np.zeros(100)
    target = np.random.default_rng(0).normal(size=100)

    m = compute_forecast_metrics(pred, target, horizon=1)

    assert m.n_obs == 100
    assert np.isnan(m.rank_ic) and np.isnan(m.rank_ic_se) and m.hac_lags is None
    assert np.isnan(m.hit_rate) and m.hit_n == 0  # zero signs have no direction


def test_below_min_obs_everything_is_nan_but_n_is_kept():
    pred, target = _fixture()

    m = compute_forecast_metrics(pred[:20], target[:20], horizon=1, min_obs=30)

    assert (m.n_obs, m.min_obs) == (20, 30)
    assert np.isnan(m.rank_ic) and np.isnan(m.rank_ic_se) and np.isnan(m.hit_rate)
    assert all(np.isnan(v) for v in m.ic_decay.values())


def test_ic_se_matches_statsmodels_hac_and_respects_the_horizon():
    """Reference: statsmodels 0.14 OLS(z(rank target), [1, z(rank pred)])
    .fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True})."""
    pred, target = _fixture()

    se, lags = ic_se_newey_west(pred, target, horizon=1)
    assert lags == 4  # newey_west_lags(120)
    assert se == pytest.approx(0.08180439044435259, rel=1e-10)

    se, lags = ic_se_newey_west(pred, target, horizon=10)
    assert lags == 9  # horizon - 1 wins
    assert se == pytest.approx(0.08299640632342516, rel=1e-10)

    assert rank_ic(pred, target) == pytest.approx(0.29772900895895554, rel=1e-12)


def test_hit_rate_leaves_out_zero_signs():
    rate, n = hit_rate([1.0, -1.0, 0.0, 2.0, 1.0], [0.5, 0.5, 0.3, 0.0, 1.0])
    assert (rate, n) == (pytest.approx(2 / 3), 3)


def test_fold_summary_uses_ddof_1_and_leaves_out_nan_folds():
    ics = [0.02, 0.06, np.nan, -0.01]

    s = compute_fold_ic_summary(ics)

    defined = pd.Series([0.02, 0.06, -0.01])
    assert (s.n_folds, s.n_undefined) == (3, 1)
    assert s.std == pytest.approx(defined.std(ddof=1))
    assert s.se == pytest.approx(defined.std(ddof=1) / np.sqrt(3))
    assert s.share_positive == pytest.approx(2 / 3)

    one = compute_fold_ic_summary([0.05])
    assert one.mean == 0.05 and np.isnan(one.std) and np.isnan(one.se)
