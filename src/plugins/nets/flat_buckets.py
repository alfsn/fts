# src/plugins/nets/flat_buckets.py

from abc import ABC, abstractmethod
from typing import Sequence

from trading_bot.core.schemas import BarData


class BaseFlatBucket(ABC):
    """Abstract base for determining if a predicted return is 'flat'."""

    @abstractmethod
    def is_flat(self, predicted_return: float, bars: Sequence[BarData]) -> bool:
        pass


class DummyFlat(BaseFlatBucket):
    """Uses a fixed threshold to determine 'flat' status."""

    def __init__(self, threshold: float = 0.001) -> None:
        self.threshold = threshold

    def is_flat(self, predicted_return: float, bars: Sequence[BarData]) -> bool:
        return abs(predicted_return) < self.threshold


class DynamicFlat(BaseFlatBucket):
    """
    Uses bid-ask spread and ATR to determine a dynamic 'flat' zone.
    Threshold = max(bid_ask_spread_pct, k * ATR_pct)
    """

    def __init__(self, k: float = 0.5, period: int = 10) -> None:
        self.k = k
        self.period = period

    def is_flat(self, predicted_return: float, bars: Sequence[BarData]) -> bool:
        if not bars or len(bars) < self.period:
            return abs(predicted_return) < 0.001  # Fallback

        # Calculate ATR (simplified for this v0: High - Low average)
        total_range = 0.0
        for bar in bars[-self.period :]:
            total_range += (bar.high - bar.low) / bar.close
        atr_pct = total_range / self.period

        # We don't have bid/ask spread directly in historical bars here,
        # but we could approximate it or assume a minimum threshold.
        # For now, let's use the ATR-based threshold.
        dynamic_threshold = self.k * atr_pct
        return abs(predicted_return) < dynamic_threshold
