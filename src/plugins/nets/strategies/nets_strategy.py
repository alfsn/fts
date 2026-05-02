# src/plugins/nets/strategies/nets_strategy.py

import logging
from typing import Any, Sequence

from trading_bot.core.schemas import IngestionEngineOutput, SignalType, TradeSignal
from trading_bot.core.transforms import BaseTransform
from trading_bot.strategy.abc import BaseStrategy

from ..flat_buckets import BaseFlatBucket

logger = logging.getLogger(__name__)


class NetsStrategy(BaseStrategy):
    """
    A strategy that uses a neural net or ML model to generate signals.
    Orchestrates transforms, inference, and flat-bucket classification.
    """

    def __init__(
        self,
        model: Any,  # Placeholder for XGBoost/CNN model
        transform: BaseTransform,
        flat_bucket: BaseFlatBucket,
        lookback_period: int = 20,
        name_suffix: str = "v1",
    ) -> None:
        self.model = model
        self.transform = transform
        self.flat_bucket = flat_bucket
        self.lookback_period = lookback_period
        self._name = f"nets_strategy_{name_suffix}"

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, data: IngestionEngineOutput) -> Sequence[TradeSignal]:
        signals = []

        for market_id, market_data in data.market_data.items():
            bars = market_data.recent_bars
            if len(bars) < self.lookback_period:
                continue

            # 1. Extract prices and transform
            prices = [b.close for b in bars[-self.lookback_period :]]
            features = self.transform.transform(prices)

            # 2. Inference (Mock for v0)
            # In reality, this would be: predicted_return = self.model.predict(features)
            # For v0, let's assume the model returns the last return + a small bias
            predicted_return = features[-1] * 1.1 if features else 0.0

            # 3. Classify using Flat Bucket
            if self.flat_bucket.is_flat(predicted_return, bars):
                signal_type = SignalType.FLAT
                confidence = 0.5
            elif predicted_return > 0:
                signal_type = SignalType.BUY
                confidence = min(abs(predicted_return) * 10, 1.0)  # Simple scaling
            else:
                signal_type = SignalType.SELL
                confidence = min(abs(predicted_return) * 10, 1.0)

            signals.append(
                TradeSignal(
                    market_id=market_id,
                    strategy_name=self.name,
                    signal_type=signal_type,
                    confidence=confidence,
                )
            )

        return signals
