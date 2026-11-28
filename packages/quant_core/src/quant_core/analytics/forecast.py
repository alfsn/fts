"""
How good is a return forecast, and how sure are we (R1): rank IC, its
Newey-West standard error, IC decay and the directional hit rate.

Inputs are equal-length array-likes in time order (prediction i is scored
against target i). Pairs with a NaN on either side are dropped, never filled.
Every function returns NaN when the answer is undefined (fewer than three
pairs, or a constant series), never 0.0.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .regression import hac_ols, newey_west_lags

MIN_PAIRS = 3


def drop_unpaired(pred, target) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten both sides and drop pairs with a NaN on either side."""
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(target, dtype=float).ravel()
    if len(p) != len(t):
        raise ValueError(f"{len(p)} predictions for {len(t)} targets.")
    keep = ~(np.isnan(p) | np.isnan(t))
    return p[keep], t[keep]


def _standardized_ranks(pred, target) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Average ranks (ties as in ``spearmanr``), centred and scaled with
    ddof=1, or None when there are too few pairs or a side is constant."""
    p, t = drop_unpaired(pred, target)
    if len(p) < MIN_PAIRS:
        return None
    rp = pd.Series(p).rank(method="average").to_numpy()
    rt = pd.Series(t).rank(method="average").to_numpy()
    sp, st = rp.std(ddof=1), rt.std(ddof=1)
    if not (sp > 0 and st > 0):
        return None
    return (rp - rp.mean()) / sp, (rt - rt.mean()) / st


def rank_ic(pred, target) -> float:
    """Spearman correlation of predictions and realised targets."""
    z = _standardized_ranks(pred, target)
    if z is None:
        return float("nan")
    zp, zt = z
    return float(zp @ zt / (len(zp) - 1))


def ic_hac_lags(n: int, horizon: int) -> int:
    """Newey-West lags for an IC: the automatic bandwidth, but at least
    ``horizon - 1``, because targets over ``horizon`` bars overlap (D12)."""
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 bar, got {horizon}.")
    return max(newey_west_lags(n), horizon - 1)


def ic_se_newey_west(pred, target, horizon: int) -> Tuple[float, Optional[int]]:
    """
    HAC standard error of the rank IC and the lags used. The rank IC is the
    slope of standardized target ranks on standardized prediction ranks, so
    its SE is that slope's Newey-West SE (``hac_ols``). (NaN, None) when the
    IC is undefined.
    """
    z = _standardized_ranks(pred, target)
    if z is None:
        return float("nan"), None
    zp, zt = z
    lags = ic_hac_lags(len(zp), horizon)
    fit = hac_ols(pd.Series(zt), pd.DataFrame({"rank_pred": zp}), lags=lags)
    return float(fit.se[1]), lags


def rank_ic_decay(pred, target, lags: int) -> Dict[int, float]:
    """Rank IC of prediction i against target i + k for k = 0..lags. The
    shift is by position, before NaN pairs are dropped."""
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(target, dtype=float).ravel()
    if len(p) != len(t):
        raise ValueError(f"{len(p)} predictions for {len(t)} targets.")
    return {k: rank_ic(p[: len(p) - k], t[k:]) for k in range(lags + 1)}


def hit_rate(pred, target) -> Tuple[float, int]:
    """
    Share of pairs whose signs agree, and how many pairs were scored. A zero
    on either side has no direction and is left out, so a forecast of zeros
    scores NaN, not 50%.
    """
    p, t = drop_unpaired(pred, target)
    sp, st = np.sign(p), np.sign(t)
    keep = (sp != 0) & (st != 0)
    n = int(keep.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(sp[keep] == st[keep])), n
