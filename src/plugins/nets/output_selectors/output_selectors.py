from typing import List, Sequence, Tuple

import numpy as np

from trading_bot.core.schemas import BarData

from ..enums import PredictionSignal
from .abc import BaseOutputSelector, BaseRegressionOutputSelector


class SimpleThresholdClassifier(BaseRegressionOutputSelector):
    """Uses a fixed threshold to determine 'flat' status and computes return-based confidence."""

    def __init__(
        self, threshold: float = 0.001, confidence_multiplier: float = 10.0
    ) -> None:
        self.threshold = threshold
        self.confidence_multiplier = confidence_multiplier

    def process_return(
        self, predicted_return: float, bars: Sequence[BarData]
    ) -> Tuple[PredictionSignal, float]:
        if abs(predicted_return) < self.threshold:
            return PredictionSignal.FLAT, 0.5

        signal = PredictionSignal.UP if predicted_return > 0 else PredictionSignal.DOWN
        confidence = min(abs(predicted_return) * self.confidence_multiplier, 1.0)
        return signal, confidence


def calculate_atr_pct(bars: Sequence[BarData], period: int) -> float:
    """
    Calculates the Average True Range percentage (ATR_pct) over the specified period,
    properly accounting for price gaps and protecting against division by zero.
    """
    if not bars or len(bars) < period:
        return 0.0

    total_tr_pct = 0.0
    for i in range(len(bars) - period, len(bars)):
        bar = bars[i]
        prev_bar = bars[i - 1] if i > 0 else bar

        high_low = bar.high - bar.low
        high_prev_close = abs(bar.high - prev_bar.close)
        low_prev_close = abs(bar.low - prev_bar.close)

        true_range = max(high_low, high_prev_close, low_prev_close)
        close_denom = (
            prev_bar.close
            if prev_bar.close > 1e-8
            else (bar.close if bar.close > 1e-8 else 1.0)
        )
        total_tr_pct += true_range / close_denom

    return total_tr_pct / period


class DynamicThresholdClassifier(BaseRegressionOutputSelector):
    """
    Uses ATR to determine a dynamic 'flat' zone and computes return-based confidence.
    Threshold = k * ATR_pct
    """

    def __init__(
        self, k: float = 0.5, period: int = 10, confidence_multiplier: float = 10.0
    ) -> None:
        self.k = k
        self.period = period
        self.confidence_multiplier = confidence_multiplier

    def process_return(
        self, predicted_return: float, bars: Sequence[BarData]
    ) -> Tuple[PredictionSignal, float]:
        if not bars or len(bars) < self.period:
            # Fallback to simple threshold
            if abs(predicted_return) < 0.001:
                return PredictionSignal.FLAT, 0.5
            signal = (
                PredictionSignal.UP if predicted_return > 0 else PredictionSignal.DOWN
            )
            confidence = min(abs(predicted_return) * self.confidence_multiplier, 1.0)
            return signal, confidence

        atr_pct = calculate_atr_pct(bars, self.period)

        dynamic_threshold = self.k * atr_pct
        if abs(predicted_return) < dynamic_threshold:
            return PredictionSignal.FLAT, 0.5

        signal = PredictionSignal.UP if predicted_return > 0 else PredictionSignal.DOWN
        confidence = min(abs(predicted_return) * self.confidence_multiplier, 1.0)
        return signal, confidence


class ClassificationOutputSelector(BaseOutputSelector):
    """
    Selects output and confidence from class probabilities.
    Assumes output is a probability vector where each element corresponds to a PredictionSignal.
    """

    def __init__(self, class_labels: List[PredictionSignal]) -> None:
        self.class_labels = class_labels

    def select_output(
        self, prediction: np.ndarray, bars: Sequence[BarData]
    ) -> Tuple[PredictionSignal, float]:
        probs = prediction.flatten()
        if len(probs) != len(self.class_labels):
            raise ValueError(
                f"Expected {len(self.class_labels)} probabilities, got {len(probs)}"
            )

        idx = int(np.argmax(probs))
        return self.class_labels[idx], float(probs[idx])


class QuantileOutputSelector(BaseOutputSelector):
    """
    Processes quantile predictions [q10, q50, q90] to determine signal and confidence.
    Confidence is based on the precision (inverse of quantile spread).
    """

    def __init__(self, threshold: float = 0.001, spread_scale: float = 1.0) -> None:
        self.threshold = threshold
        self.spread_scale = spread_scale

    def select_output(
        self, prediction: np.ndarray, bars: Sequence[BarData]
    ) -> Tuple[PredictionSignal, float]:
        quantiles = prediction.flatten()
        if len(quantiles) != 3:
            raise ValueError(
                f"Expected 3 quantiles [q10, q50, q90], got {len(quantiles)}"
            )

        q10, q50, q90 = quantiles

        # Direction based on median (q50)
        if abs(q50) < self.threshold:
            signal = PredictionSignal.FLAT
        elif q50 > 0:
            signal = PredictionSignal.UP
        else:
            signal = PredictionSignal.DOWN

        # Confidence is high if the spread (q90 - q10) is narrow (lower uncertainty)
        spread = max(q90 - q10, 1e-8)
        confidence = float(np.exp(-spread * self.spread_scale))

        return signal, confidence
