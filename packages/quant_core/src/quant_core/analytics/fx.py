from typing import TypeVar

import numpy as np
import pandas as pd

PriceT = TypeVar("PriceT", float, pd.Series)


def implied_ccl(local_price: PriceT, adr_price: PriceT, ratio: float) -> PriceT:
    """
    CCL ("contado con liquidación") rate implied by a local listing and its ADR,
    in local currency per USD: `local * ratio / adr`, where `ratio` is local shares
    per ADR *on the date of the prices* (GGAL 10; YPF 1 before its 2026-08-03
    local split, 10 since). Series are aligned on their index; non-positive or
    missing prices give NaN.
    """
    if isinstance(local_price, pd.Series) or isinstance(adr_price, pd.Series):
        local, adr = pd.Series(local_price).align(pd.Series(adr_price), join="inner")
        local, adr = local.where(local > 0), adr.where(adr > 0)
        return local * ratio / adr
    if not (local_price > 0 and adr_price > 0):
        return np.nan
    return local_price * ratio / adr_price


def implied_ccl_frame(
    closes: pd.DataFrame, pairs: tuple[tuple[str, str, float], ...]
) -> pd.DataFrame:
    """The CCL each pair implies on its own, one column per pair available in
    ``closes`` ("LOCAL/ADR"); empty when none is. Same inputs as
    ``ccl_from_pairs`` (SPLIT-adjusted closes)."""
    implied = {
        f"{local}/{adr}": implied_ccl(closes[local], closes[adr], ratio)
        for local, adr, ratio in pairs
        if local in closes and adr in closes
    }
    return pd.DataFrame(implied) if implied else pd.DataFrame()


def ccl_from_pairs(
    closes: pd.DataFrame,
    pairs: tuple[tuple[str, str, float], ...],
    max_divergence: float = 0.05,
) -> pd.Series:
    """
    Daily CCL (local currency per USD) from the `pairs` (local, ADR, local
    shares per ADR) available in `closes` (columns = symbols): the mean of the
    implied rates, or NaN on days where they disagree by more than
    `max_divergence` (max/min - 1). A disagreement that large means a wrong
    ratio or bad print, not noise, and averaging it in would be silent. Days
    with a single pair use it unchecked.

    The pairs' ratios are today's, so pass SPLIT-adjusted closes (`adjust`);
    the reference pairs live in `quant_data.reference.ARS_CCL_PAIRS`.
    """
    rates = implied_ccl_frame(closes, pairs)
    if rates.empty:
        return pd.Series(dtype=float)
    divergence = rates.max(axis=1) / rates.min(axis=1) - 1
    return rates.mean(axis=1).where(~(divergence > max_divergence))
