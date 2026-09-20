# src/trading_bot/data_ingestion/bars.py

import math
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Literal, Optional, overload

from ..core.enums import BarType
from ..core.schemas import BarData, Trade


class BaseBarAggregator(ABC):
    """
    Abstract Base Class for bar aggregation logic.
    Follows the SOLID principles by providing a clear interface for
    different types of bar aggregation (Time, Volume, Dollar).
    """

    def __init__(self, market_id: str, bar_type: BarType) -> None:
        self.market_id = market_id
        self.bar_type = bar_type
        self.current_bar: Optional[BarData] = None

    @abstractmethod
    def add_trade(self, trade: Trade) -> Optional[BarData]:
        """
        Adds a trade to the current aggregator and returns a completed bar if
        the aggregation criteria (time, volume, or dollar) is met.
        """
        pass

    def _initialize_bar(self, trade: Trade, bar_timestamp: datetime) -> None:
        """Initializes a new BarData object."""
        self.current_bar = BarData(
            timestamp=bar_timestamp,
            open=trade.price,
            high=trade.price,
            low=trade.price,
            close=trade.price,
            volume=trade.size,
            bar_type=self.bar_type,
            ticks_count=1,
            dollar_volume=trade.price * trade.size,
        )

    def _update_bar(self, trade: Trade) -> None:
        """Updates the current bar with new trade data."""
        if not self.current_bar:
            return
        self.current_bar.high = max(self.current_bar.high, trade.price)
        self.current_bar.low = min(self.current_bar.low, trade.price)
        self.current_bar.close = trade.price
        self.current_bar.volume += trade.size
        self.current_bar.ticks_count += 1
        self.current_bar.dollar_volume += trade.price * trade.size

    def _add_trade_threshold(
        self, trade: Trade, metric_name: str, threshold: float
    ) -> Optional[BarData]:
        """
        Generic helper for threshold-based aggregators (volume or dollar).
        """
        if not self.current_bar:
            self._initialize_bar(trade, trade.timestamp)
        else:
            self._update_bar(trade)

        if self.current_bar and getattr(self.current_bar, metric_name) >= threshold:
            completed_bar = self.current_bar
            completed_bar.timestamp = trade.timestamp
            self.current_bar = None
            return completed_bar

        return None


class TimeBarAggregator(BaseBarAggregator):
    """
    Aggregates trades into time-based bars (e.g., 5-minute, 1-hour, Daily).
    """

    def __init__(self, market_id: str, interval: timedelta) -> None:
        super().__init__(market_id, BarType.TIME)
        self.interval = interval
        self.next_bar_boundary: Optional[datetime] = None

    def add_trade(self, trade: Trade) -> Optional[BarData]:
        completed_bar = None

        if self.next_bar_boundary and trade.timestamp >= self.next_bar_boundary:
            completed_bar = self.current_bar
            self.current_bar = None
            self.next_bar_boundary = None

        if not self.current_bar:
            # Align to the interval boundary
            ts_seconds = trade.timestamp.timestamp()
            interval_seconds = self.interval.total_seconds()

            # Boundary is the end of the interval
            boundary_seconds = (
                math.floor(ts_seconds / interval_seconds) + 1
            ) * interval_seconds
            self.next_bar_boundary = datetime.fromtimestamp(
                boundary_seconds, tz=trade.timestamp.tzinfo
            )

            self._initialize_bar(trade, self.next_bar_boundary)
        else:
            self._update_bar(trade)

        return completed_bar


class VolumeBarAggregator(BaseBarAggregator):
    """
    Aggregates trades into bars based on a fixed volume threshold.
    """

    def __init__(self, market_id: str, volume_threshold: float) -> None:
        super().__init__(market_id, BarType.VOLUME)
        self.threshold = volume_threshold

    def add_trade(self, trade: Trade) -> Optional[BarData]:
        return self._add_trade_threshold(trade, "volume", self.threshold)


class DollarBarAggregator(BaseBarAggregator):
    """
    Aggregates trades into bars based on a fixed dollar volume (PxQ) threshold.
    """

    def __init__(self, market_id: str, dollar_threshold: float) -> None:
        super().__init__(market_id, BarType.DOLLAR)
        self.threshold = dollar_threshold

    def add_trade(self, trade: Trade) -> Optional[BarData]:
        return self._add_trade_threshold(trade, "dollar_volume", self.threshold)


class BarFactory:
    """
    Factory for creating BarAggregators based on BarType.
    This allows for an extensible "Internal Factory" without hardcoding
    all possible bar variations in the main logic.
    """

    @overload
    @staticmethod
    def create_aggregator(
        bar_type: Literal[BarType.TIME], market_id: str, *, interval: timedelta
    ) -> TimeBarAggregator: ...

    @overload
    @staticmethod
    def create_aggregator(
        bar_type: Literal[BarType.VOLUME], market_id: str, *, threshold: float
    ) -> VolumeBarAggregator: ...

    @overload
    @staticmethod
    def create_aggregator(
        bar_type: Literal[BarType.DOLLAR], market_id: str, *, threshold: float
    ) -> DollarBarAggregator: ...

    @staticmethod
    def create_aggregator(
        bar_type: BarType, market_id: str, **kwargs: Any
    ) -> BaseBarAggregator:

        if bar_type == BarType.TIME:
            if "interval" not in kwargs:
                raise ValueError("Time bars require an 'interval' (timedelta).")
            return TimeBarAggregator(market_id, kwargs["interval"])
        elif bar_type == BarType.VOLUME:
            if "threshold" not in kwargs:
                raise ValueError("Volume bars require a 'threshold' (float).")
            return VolumeBarAggregator(market_id, kwargs["threshold"])
        elif bar_type == BarType.DOLLAR:
            if "threshold" not in kwargs:
                raise ValueError("Dollar bars require a 'threshold' (float).")
            return DollarBarAggregator(market_id, kwargs["threshold"])
        else:
            raise ValueError(f"Unsupported bar type: {bar_type}")
