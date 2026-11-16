from typing import TypeVar

import numpy as np
import pandas as pd

PriceT = TypeVar("PriceT", float, pd.Series)


def implied_ccl(local_price: PriceT, adr_price: PriceT, ratio: float) -> PriceT:
    """
    CCL ("contado con liquidación") rate implied by a local listing and its ADR,
    in local currency per USD: `local * ratio / adr`, where `ratio` is local shares
    per ADR (GGAL 10, YPF 1). Series are aligned on their index; non-positive or
    missing prices give NaN.
    """
    if isinstance(local_price, pd.Series) or isinstance(adr_price, pd.Series):
        local, adr = pd.Series(local_price).align(pd.Series(adr_price), join="inner")
        local, adr = local.where(local > 0), adr.where(adr > 0)
        return local * ratio / adr
    if not (local_price > 0 and adr_price > 0):
        return np.nan
    return local_price * ratio / adr_price


# (local listing, USD ADR, local shares per ADR) used to imply ARS CCL.
# Ratios as quoted by Yahoo; checked against each other and the official rate
# over 2021-2026 (GGAL.BA/GGAL and YPFD.BA/YPF both imply 10:1).
ARS_CCL_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("GGAL.BA", "GGAL", 10.0),
    ("YPFD.BA", "YPF", 10.0),
)


def ccl_from_pairs(
    closes: pd.DataFrame,
    pairs: tuple[tuple[str, str, float], ...] = ARS_CCL_PAIRS,
    max_divergence: float = 0.05,
) -> pd.Series:
    """
    Daily CCL (local currency per USD) from the pairs available in `closes`
    (columns = symbols): the mean of the implied rates, or NaN on days where
    they disagree by more than `max_divergence` (max/min - 1). A disagreement
    that large means a wrong ratio or bad print, not noise, and averaging it in
    would be silent. Days with a single pair use it unchecked.
    """
    implied = [
        implied_ccl(closes[local], closes[adr], ratio)
        for local, adr, ratio in pairs
        if local in closes and adr in closes
    ]
    if not implied:
        return pd.Series(dtype=float)
    rates = pd.concat(implied, axis=1)
    divergence = rates.max(axis=1) / rates.min(axis=1) - 1
    return rates.mean(axis=1).where(~(divergence > max_divergence))
