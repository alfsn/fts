# src/plugins/nets/training.py

import logging
from typing import Any, Sequence

from trading_bot.core.schemas import BarData
from trading_bot.core.training import BaseModelTrainer

logger = logging.getLogger(__name__)


class XGBoostTrainer(BaseModelTrainer):
    """
    Placeholder for an XGBoost-based trainer.
    """

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        logger.info(f"Starting XGBoost training on {len(historical_bars)} bars.")
        # V2: Implement actual training logic
        return "mock_xgboost_model"


class CNNTrainer(BaseModelTrainer):
    """
    Placeholder for a CNN-based trainer.
    """

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        logger.info(f"Starting CNN training on {len(historical_bars)} bars.")
        # V2: Implement actual training logic
        return "mock_cnn_model"
