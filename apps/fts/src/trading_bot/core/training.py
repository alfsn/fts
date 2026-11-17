# src/trading_bot/core/training.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

from .schemas import BarData

logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """
    Interface for asynchronous model training.
    Implementations should handle data fetching, training, and artifact saving.
    """

    @abstractmethod
    def train(self, historical_bars: Sequence[BarData]) -> Any:
        """
        Trains a model on the provided data and returns the trained artifact.
        """
        pass
