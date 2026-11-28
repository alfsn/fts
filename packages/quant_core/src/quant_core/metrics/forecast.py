"""Forecast scores (R1): one record per prediction set, one per set of folds."""

from typing import Sequence

import numpy as np
import pandas as pd

from ..analytics import (
    drop_unpaired,
    hit_rate,
    ic_se_newey_west,
    rank_ic,
    rank_ic_decay,
)
from ..models import FoldICSummary, ForecastMetrics

# Validation sets are about 100 points (20% of ~500 bars), so the bar is lower
# than MIN_SAMPLE_OBS; the SE beside the IC says how noisy it is (R1-D3).
MIN_FORECAST_OBS = 30


def compute_forecast_metrics(
    pred,
    target,
    horizon: int,
    min_obs: int = MIN_FORECAST_OBS,
    decay_lags: int = 5,
) -> ForecastMetrics:
    """Score predictions against realised targets (time order). Below
    ``min_obs`` pairs every score is NaN."""
    p, t = drop_unpaired(pred, target)
    n = len(p)
    rate, hit_n = hit_rate(p, t)
    base = dict(n_obs=n, min_obs=min_obs, horizon=horizon, hit_n=hit_n)
    if n < min_obs:
        return ForecastMetrics(
            **base, ic_decay={k: float("nan") for k in range(decay_lags + 1)}
        )
    se, lags = ic_se_newey_west(p, t, horizon)
    return ForecastMetrics(
        **base,
        rank_ic=rank_ic(p, t),
        rank_ic_se=se,
        hac_lags=lags,
        hit_rate=rate if hit_n >= min_obs else float("nan"),
        ic_decay=rank_ic_decay(pred, target, decay_lags),
    )


def compute_fold_ic_summary(fold_ics: Sequence[float]) -> FoldICSummary:
    """Mean fold IC with SE = std(ddof=1) / sqrt(folds), plus median, IR and
    the share of positive folds."""
    ics = pd.Series(list(fold_ics), dtype=float)
    defined = ics.dropna()
    n = len(defined)
    if n == 0:
        return FoldICSummary(n_undefined=len(ics))
    std = float(defined.std(ddof=1)) if n > 1 else float("nan")
    mean = float(defined.mean())
    return FoldICSummary(
        n_folds=n,
        n_undefined=len(ics) - n,
        mean=mean,
        median=float(defined.median()),
        std=std,
        se=std / np.sqrt(n),
        ir=mean / std if std > 0 else float("nan"),
        share_positive=float((defined > 0).mean()),
    )
