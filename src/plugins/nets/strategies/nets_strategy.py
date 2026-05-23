# src/plugins/nets/strategies/nets_strategy.py

import logging
from typing import Sequence

import numpy as np

from trading_bot.core.schemas import IngestionEngineOutput, SignalType, TradeSignal
from trading_bot.core.transforms import BaseTransform
from trading_bot.strategy.abc import BaseStrategy

from ..classifiers import BaseClassifier
from ..enums import PredictionSignal
from ..inference import ONNXPredictor

logger = logging.getLogger(__name__)


class NetsStrategy(BaseStrategy):
    """
    A strategy that uses an ONNX model to generate forecasts and classifies
    them into UP, FLAT, or DOWN signals.
    """

    def __init__(
        self,
        predictor: ONNXPredictor,
        transform: BaseTransform,
        classifier: BaseClassifier,
        lookback_period: int = 20,
        name_suffix: str = "v1",
    ) -> None:
        self.predictor = predictor
        self.transform = transform
        self.classifier = classifier
        self.lookback_period = lookback_period
        self._name = f"nets_strategy_{name_suffix}"

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, data: IngestionEngineOutput) -> Sequence[TradeSignal]:
        signals = []

        for market_id, market_data in data.market_data.items():
            bars = market_data.recent_bars
            if len(bars) < self.lookback_period + 1:
                continue

            # 1. Extract prices and transform
            prices = [b.close for b in bars[-self.lookback_period - 1 :]]
            features = self.transform.transform(np.array(prices).reshape(-1, 1))

            # 2. Inference via ONNX
            # Ensure features are in the correct shape
            # (1, lookback) for the pipeline/model
            input_data = features.flatten().reshape(1, -1).astype(np.float32)
            prediction = self.predictor.predict(input_data)
            predicted_return = float(prediction.flatten()[0])

            # 3. Classify using the Classifier (UFD)
            signal_direction = self.classifier.classify(predicted_return, bars)

            # 4. Map UFD to TradeSignal
            if signal_direction == PredictionSignal.FLAT:
                signal_type = SignalType.FLAT
                confidence = 0.5
            elif signal_direction == PredictionSignal.UP:
                signal_type = SignalType.BUY
                confidence = min(abs(predicted_return) * 10, 1.0)
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
