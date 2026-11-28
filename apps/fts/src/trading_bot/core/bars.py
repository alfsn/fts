"""
fts's one way to read research bars from the price cache (M2a).

The cache belongs to ``quant_data`` (``MarketDataRepository``); this module
only adds fts's policy on top. fts trains and backtests on raw prices, and a
raw series across a split shows a fake crash (a 4:1 split reads as -75%), so a
window that contains a split is refused instead of loaded (M2-D2).
Dividends are not checked: a known limit (backlog "Split- and
dividend-adjusted research").
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict
from quant_core.analytics import split_factor
from quant_core.models import BarData
from quant_data.db import MarketDataRepository


class UnadjustedSplitError(ValueError):
    """A split falls inside a window of raw bars."""


class ResearchBars(BaseModel):
    """Warm-up bars (strictly before the start) and the window's bars."""

    model_config = ConfigDict(extra="forbid")

    warmup: List[BarData]
    bars: List[BarData]


def check_no_split(
    repo: MarketDataRepository, instrument_id: str, bars: List[BarData]
) -> None:
    """Raise ``UnadjustedSplitError`` if a cached split falls inside ``bars``."""
    if len(bars) < 2:
        return
    actions = repo.load_corporate_actions(instrument_id)
    if not actions:
        return
    index = pd.DatetimeIndex([b.timestamp for b in bars])
    factor = split_factor(index, actions).to_numpy()
    changes = (factor[1:] != factor[:-1]).nonzero()[0]
    if len(changes):
        i = changes[0] + 1
        raise UnadjustedSplitError(
            f"{instrument_id}: a {factor[i - 1] / factor[i]:g}:1 split takes effect "
            f"on {bars[i].timestamp.date()}, inside {bars[0].timestamp.date()}.."
            f"{bars[-1].timestamp.date()}. Prices are raw (as traded), so the "
            "split would read as a crash. Pick a window after the split."
        )


def load_research_bars(
    repo: MarketDataRepository,
    instrument_id: str,
    interval: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    warmup: int = 0,
) -> ResearchBars:
    """
    Raw bars of one series for training or a backtest. ``warmup`` bars before
    ``start`` come with it; the split check covers warm-up and window.
    """
    warmup_bars = (
        repo.load_bars_before(instrument_id, interval, start, warmup)
        if warmup > 0 and start is not None
        else []
    )
    bars = repo.load_bars(instrument_id, interval, start, end)
    check_no_split(repo, instrument_id, warmup_bars + bars)
    return ResearchBars(warmup=warmup_bars, bars=bars)
