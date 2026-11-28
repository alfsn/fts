"""
Bar intervals and the rule for when a hole in a bar series is a gap (M2-D5).

A venue's trading hours decide what a gap is: a 24/7 crypto venue has a bar
every interval, so five missing bars is a gap; an exchange closes overnight
and at weekends, so short bars are expected to pause for days. Each data
vendor declares its rule (``BaseMarketDataProvider.gap_rule``).
"""

from abc import ABC, abstractmethod
from datetime import timedelta

_UNITS = {
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
    "w": timedelta(weeks=1),
    "wk": timedelta(weeks=1),
    "mo": timedelta(days=30),  # yfinance's month; used for gap checks only
    "M": timedelta(days=30),  # ccxt's month
}


def interval_to_timedelta(interval: str) -> timedelta:
    """'1m', '4h', '1d', '1w' (yfinance '1wk'/'1mo', ccxt '1M') as a duration."""
    if not interval:
        raise ValueError("Interval string cannot be empty.")
    digits = len(interval) - len(interval.lstrip("0123456789"))
    value, unit = interval[:digits], interval[digits:]
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
