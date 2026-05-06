from abc import ABC, abstractmethod
from typing import Sequence

from trading_bot.core.schemas import BarData

from .enums import PredictionSignal


class BaseClassifier(ABC):
    """Abstract base for classifying a predicted return into UFD signals."""

    @abstractmethod
    def classify(
        self, predicted_return: float, bars: Sequence[BarData]
    ) -> PredictionSignal:
        pass


class SimpleThresholdClassifier(BaseClassifier):
    """Uses a fixed threshold to determine 'flat' status."""

    def __init__(self, threshold: float = 0.001) -> None:
        self.threshold = threshold

    def classify(
        self, predicted_return: float, bars: Sequence[BarData]
    ) -> PredictionSignal:
        if abs(predicted_return) < self.threshold:
            return PredictionSignal.FLAT
        return PredictionSignal.UP if predicted_return > 0 else PredictionSignal.DOWN


class DynamicThresholdClassifier(BaseClassifier):
    """
    Uses ATR to determine a dynamic 'flat' zone.
    Threshold = k * ATR_pct
    """

    def __init__(self, k: float = 0.5, period: int = 10) -> None:
        self.k = k
        self.period = period

    def classify(
        self, predicted_return: float, bars: Sequence[BarData]
    ) -> PredictionSignal:
        if not bars or len(bars) < self.period:
            # Fallback to simple threshold
            if abs(predicted_return) < 0.001:
                return PredictionSignal.FLAT
            return (
                PredictionSignal.UP if predicted_return > 0 else PredictionSignal.DOWN
            )

        # Calculate ATR (simplified: High - Low average)
        total_range = 0.0
        for bar in bars[-self.period :]:
            total_range += (bar.high - bar.low) / bar.close
        atr_pct = total_range / self.period

        dynamic_threshold = self.k * atr_pct
        if abs(predicted_return) < dynamic_threshold:
            return PredictionSignal.FLAT
        return PredictionSignal.UP if predicted_return > 0 else PredictionSignal.DOWN
