import json
import logging
from typing import Optional, Sequence

import numpy as np

from trading_bot.core.dataset import DatasetBuilder
from trading_bot.core.schemas import IngestionEngineOutput, SignalType, TradeSignal
from trading_bot.core.transforms import BaseTransform
from trading_bot.strategy.abc import BaseStrategy

from ..enums import PredictionSignal
from ..inference import ONNXPredictor
from ..output_selectors import BaseOutputSelector

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
        allow_in_sample: bool = False,
    ) -> None:
        self.predictor = predictor
        self.transform = transform
        self.output_selector = output_selector
        self.lookback_period = lookback_period
        self.feature_cols = list(feature_cols) if feature_cols else ["close"]
        self.allow_in_sample = allow_in_sample
        self._name = f"nets_strategy_{name_suffix}"

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self, data: IngestionEngineOutput, db: Optional[object] = None
    ) -> Sequence[TradeSignal]:
        # Check lookahead guardrail
        if self.predictor.model_metadata:
            self.predictor.model_metadata.validate_timestamp(
                timestamp=data.timestamp,
                allow_in_sample=self.allow_in_sample,
                strategy_name=self.name,
            )

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

            # Serialize model prediction output to list then JSON string
            prediction_json = json.dumps(prediction.tolist())

            signal = TradeSignal(
                market_id=market_id,
                strategy_name=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=data.timestamp,
                prediction_output=prediction_json,
            )

            signals.append(signal)

        return signals
