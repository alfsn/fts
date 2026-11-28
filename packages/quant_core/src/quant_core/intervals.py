"""
Bar intervals and the rule for when a hole in a bar series is a gap (M2-D5).

A venue's trading hours decide what a gap is: a 24/7 crypto venue has a bar
every interval, so five missing bars is a gap; an exchange closes overnight
and at weekends, so short bars are expected to pause for days. Each data
vendor declares its rule (``BaseMarketDataProvider.gap_rule``).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

_UNITS = {
    "s": timedelta(seconds=1),
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
    "w": timedelta(weeks=1),
    "wk": timedelta(weeks=1),
    "mo": timedelta(days=30),  # yfinance's month; used for gap checks only
    "M": timedelta(days=30),  # ccxt's month
}


def interval_to_timedelta(interval: str) -> timedelta:
    """'1s', '1m', '4h', '1d', '1w' (yfinance '1wk'/'1mo', ccxt '1M') as a
    duration. Units are case-insensitive ('1H', '1D') except 'M', which is
    ccxt's month, never a minute."""
    if not interval:
        raise ValueError("Interval string cannot be empty.")
    digits = len(interval) - len(interval.lstrip("0123456789"))
    value, unit = interval[:digits], interval[digits:]
    if unit not in _UNITS:
        unit = unit.lower()
    if not value or unit not in _UNITS:
        raise ValueError(f"Unknown interval '{interval}'")
    return int(value) * _UNITS[unit]


class GapRule(ABC):
    """The longest hole a cached series of ``interval`` bars may have."""

    @abstractmethod
    def max_gap(self, interval: str) -> timedelta:
        pass


class BarGap(GapRule):
    """24/7 venues (crypto): more than ``bars`` missing intervals is a gap."""

    def __init__(self, bars: int = 5) -> None:
        self.bars = bars

    def max_gap(self, interval: str) -> timedelta:
        return self.bars * interval_to_timedelta(interval)


class SessionGap(GapRule):
    """
    Exchange-hours venues: ``bars`` intervals, but never less than
    ``min_gap``, so nights, weekends and long holidays are not gaps.
    """

    def __init__(self, bars: int = 5, min_gap: timedelta = timedelta(days=5)) -> None:
        self.bars = bars
        self.min_gap = min_gap

    def max_gap(self, interval: str) -> timedelta:
        return max(self.bars * interval_to_timedelta(interval), self.min_gap)


def find_gaps(timestamps: Iterable[datetime], max_gap: timedelta) -> pd.DataFrame:
    """
    The holes in a sorted bar series: each pair of consecutive timestamps more
    than ``max_gap`` apart (exactly ``max_gap`` is not a gap). Columns
    ``start`` (last bar before the hole), ``end`` (first bar after it) and
    ``length``; empty when there are none. The one home of the gap rule's
    comparison (M2-D5): the cache refetch, ingest and the workbench use it.
    """
    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))
    if len(index) < 2:
        return pd.DataFrame(
            {"start": index[:0], "end": index[:0], "length": pd.to_timedelta([])}
        )
    length = pd.Series(index[1:] - index[:-1])
    holes = (length > max_gap).to_numpy()
    return pd.DataFrame(
        {
            "start": index[:-1][holes],
            "end": index[1:][holes],
            "length": length[holes].to_numpy(),
        }
    )
