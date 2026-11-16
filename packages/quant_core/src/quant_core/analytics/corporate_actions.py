"""
Corporate-action maths (PR 2a).

The cache stores prices **as traded** (raw) and a list of corporate actions;
every other basis is derived on read. Two directions:

- ``unadjust_splits`` / ``unadjust_dividends`` undo a vendor's retroactive split
  adjustment so adapters can hand out raw data (Yahoo split-adjusts closes,
  volumes and dividend amounts).
- ``adjust`` turns raw closes into split-adjusted or total-return series at
  read time, CRSP-style.

Dates are compared as **calendar dates in the exchange's local frame**: any
time zone is dropped and the timestamp normalised before comparing. Adapters
call these functions on the vendor frame while it is still in the exchange's
zone, and the cache keys daily bars and ex-dates by that local date.
"""

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..enums import CorporateActionType, PriceAdjustment
from ..models import CorporateAction

PRICE_COLUMNS = ("Open", "High", "Low", "Close")
VOLUME_COLUMN = "Volume"


def local_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Calendar dates of ``index`` in its own zone (tz dropped, normalised)."""
    index = pd.DatetimeIndex(index)
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def local_ex_date(action: CorporateAction) -> pd.Timestamp:
    return local_dates(pd.DatetimeIndex([action.ex_date]))[0]


def _one_instrument(actions: Iterable[CorporateAction]) -> list[CorporateAction]:
    actions = list(actions)
    if len({a.instrument_id for a in actions}) > 1:
        raise ValueError("Corporate actions of more than one instrument were passed.")
    return actions


def _splits(actions: Iterable[CorporateAction]) -> list[CorporateAction]:
    return [a for a in actions if a.action_type is CorporateActionType.SPLIT]


def _dividends(actions: Iterable[CorporateAction]) -> list[CorporateAction]:
    return [a for a in actions if a.action_type is CorporateActionType.DIVIDEND]


def split_factor(
    index: pd.DatetimeIndex, actions: Sequence[CorporateAction]
) -> pd.Series:
    """
    For each date t, the product of ``ratio`` over splits with ex-date > t:
    how many of today's shares one share held at t has become. 1.0 on and
    after the last ex-date. Non-split actions are ignored.
    """
    dates = local_dates(index)
    factor = np.ones(len(dates))
    for split in _splits(_one_instrument(actions)):
        factor[dates < local_ex_date(split)] *= split.ratio
    return pd.Series(factor, index=index)


def unadjust_splits(
    df: pd.DataFrame,
    actions: Sequence[CorporateAction],
    price_columns: Sequence[str] = PRICE_COLUMNS,
    volume_column: str = VOLUME_COLUMN,
) -> pd.DataFrame:
    """
    Undo a vendor's split adjustment: raw price = adjusted x factor, raw
    volume = adjusted / factor. Columns not named are returned untouched.
    """
    factor = split_factor(df.index, actions)
    raw = df.copy()
    for column in price_columns:
        if column in raw:
            raw[column] = raw[column] * factor
    if volume_column in raw:
        raw[volume_column] = raw[volume_column] / factor
    return raw


def unadjust_dividends(
    dividends: pd.Series, actions: Sequence[CorporateAction]
) -> pd.Series:
    """Split-adjusted per-share amounts -> amounts per share as paid."""
    return dividends * split_factor(dividends.index, actions)


def adjust(
    closes: pd.Series, actions: Sequence[CorporateAction], mode: PriceAdjustment
) -> pd.Series:
    """
    Present raw closes of one instrument in the requested basis.

    SPLIT: raw / split factor. TOTAL_RETURN: split-adjust first, then
    multiply every price before each dividend's ex-date by ``1 - D / P_prev``,
    where ``D`` is the dividend in today's share units and ``P_prev`` the
    split-adjusted close of the last session before the ex-date (what Yahoo's
    ``Adj Close`` does). A dividend with no earlier close, or with an ex-date
    after the last close (it would scale every price equally), changes nothing.
    NaN closes stay NaN. ``D >= P_prev`` is a data error and raises.
    """
    actions = _one_instrument(actions)
    if mode is PriceAdjustment.RAW:
        return closes.copy()

    factor = split_factor(closes.index, actions)
    adjusted = closes / factor
    if mode is PriceAdjustment.SPLIT:
        return adjusted

    dates = local_dates(closes.index)
    values = adjusted.to_numpy(dtype=float).copy()
    for dividend in sorted(_dividends(actions), key=local_ex_date):
        ex = local_ex_date(dividend)
        if ex > dates[-1]:
            continue
        before = dates < ex
        earlier = values[before]
        earlier = earlier[~np.isnan(earlier)]
        if not len(earlier):
            continue
        p_prev = earlier[-1]
        d = dividend.amount / split_factor(pd.DatetimeIndex([ex]), actions).iloc[0]
        if not d < p_prev:
            raise ValueError(
                f"{dividend.instrument_id}: dividend {d} on {ex.date()} is not "
                f"below the previous close {p_prev}."
            )
        values[before] *= 1 - d / p_prev
    return pd.Series(values, index=closes.index, name=closes.name)


def adjust_frame(
    closes: pd.DataFrame,
    actions_by_symbol: Mapping[str, Sequence[CorporateAction]],
    mode: PriceAdjustment,
) -> pd.DataFrame:
    """``adjust`` column by column; a symbol without actions is returned as is."""
    return pd.DataFrame(
        {
            symbol: adjust(closes[symbol], actions_by_symbol.get(symbol, ()), mode)
            for symbol in closes.columns
        },
        index=closes.index,
    )
