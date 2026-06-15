# src/plugins/nets/output_selectors/abc.py

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from trading_bot.core.schemas import BarData

from ..enums import PredictionSignal


class BaseOutputSelector(ABC):
    """Abstract base for converting model predictions into signals and confidence."""

    @abstractmethod
    def select_output(
        self, prediction: np.ndarray, bars: Sequence[BarData]
    ) -> Tuple[PredictionSignal, float]:
        """
        Processes model prediction and returns a tuple of (PredictionSignal, confidence).
        """
        pass


class BaseRegressionOutputSelector(BaseOutputSelector, ABC):
    """Abstract base for converting regression return predictions to signals and confidence."""

    def select_output(
        self, prediction: np.ndarray, bars: Sequence[BarData]
    ) -> Tuple[PredictionSignal, float]:
        predicted_return = float(prediction.flatten()[0])
        return self.process_return(predicted_return, bars)

    @abstractmethod
    def process_return(
        self, predicted_return: float, bars: Sequence[BarData]
    ) -> Tuple[PredictionSignal, float]:
        """Calculate signal and confidence from raw expected return."""
        pass
