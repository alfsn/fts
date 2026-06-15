# src/plugins/nets/strategies/nets_strategy.py

import logging
from typing import Optional, Sequence

import numpy as np

from trading_bot.core.dataset import DatasetBuilder
from trading_bot.core.schemas import IngestionEngineOutput, SignalType, TradeSignal
from trading_bot.core.transforms import BaseTransform
from trading_bot.strategy.abc import BaseStrategy

from ..classifiers import BaseOutputSelector
from ..enums import PredictionSignal
from ..inference import ONNXPredictor

logger = logging.getLogger(__name__)


class NetsStrategy(BaseStrategy):
    """
    A strategy that uses an ONNX model to generate forecasts and classifies
    them into signals and confidence scores using an output selector.
    """

    def __init__(
        self,
        predictor: ONNXPredictor,
        transform: BaseTransform,
        output_selector: BaseOutputSelector,
        lookback_period: int = 20,
        name_suffix: str = "v1",
        feature_cols: Optional[Sequence[str]] = None,
    ) -> None:
        self.predictor = predictor
        self.transform = transform
        self.output_selector = output_selector
        self.lookback_period = lookback_period
        self.feature_cols = list(feature_cols) if feature_cols else ["close"]
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

            # 1. Extract features dynamically using DatasetBuilder
            raw_features = DatasetBuilder.to_matrix(
                bars[-self.lookback_period - 1 :],
                feature_cols=self.feature_cols,
            )
            features = self.transform.transform(raw_features)

            # 2. Inference via ONNX
            prediction = self.predictor.predict(features)

            # 3. Classify using the Output Selector
            feature_bars = bars[-self.lookback_period - 1 :]
            signal_direction, confidence = self.output_selector.select_output(
                prediction, feature_bars
            )

            # 4. Map PredictionSignal to TradeSignal
            if signal_direction == PredictionSignal.FLAT:
                signal_type = SignalType.FLAT
            elif signal_direction == PredictionSignal.UP:
                signal_type = SignalType.BUY
            else:
                signal_type = SignalType.SELL

            signals.append(
                TradeSignal(
                    market_id=market_id,
                    strategy_name=self.name,
                    signal_type=signal_type,
                    confidence=confidence,
                )
            )

        return signals
