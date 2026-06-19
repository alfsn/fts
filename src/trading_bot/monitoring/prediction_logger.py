# src/trading_bot/monitoring/prediction_logger.py
import logging
from datetime import timezone
from typing import Any, Optional, Type

from sqlalchemy.orm import Session

from ..core.models import ModelPredictionLog
from ..core.schemas import TradeSignal
from ..strategy.abc import PredictionObserver

logger = logging.getLogger(__name__)


class DatabasePredictionLogger(PredictionObserver):
    """
    Concrete implementation of PredictionObserver that logs prediction signals
    into a database session.
    """

    def __init__(
        self,
        db: Session,
        commit: bool = True,
        model_class: Type[Any] = ModelPredictionLog,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Initializes the database prediction logger.

        :param db: The SQLAlchemy DB Session.
        :param commit: If True, commits the transaction immediately.
        :param model_class: The ORM model class to log to (ModelPredictionLog or BacktestPredictionLog).
        :param run_id: The run identifier (for live/paper or backtesting prediction logging).
        """
        self.db = db
        self.commit = commit
        self.model_class = model_class
        self.run_id = run_id

    def on_prediction(self, signal: TradeSignal) -> None:
        """
        Logs a generated TradeSignal to the database. Overwrites an existing prediction
        if one already exists for the same timestamp, market, strategy, and run_id.
        """
        try:
            # Normalize timestamp to naive UTC to prevent timezone mismatches in SQLite
            ts = signal.timestamp
            ts_normalized = (
                ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts
            )

            run_id_val = self.run_id if self.run_id is not None else "live"

            # Build query filters using the unified run_id attribute
            filters = {
                "timestamp": ts_normalized,
                "market_id": signal.market_id,
                "strategy_name": signal.strategy_name,
                "run_id": run_id_val,
            }

            # Check for existing log to prevent duplicate prediction entries on re-runs
            log_entry = self.db.query(self.model_class).filter_by(**filters).first()

            if log_entry:
                log_entry.prediction_output = signal.prediction_output
                log_entry.predicted_signal = signal.signal_type.value
                log_entry.confidence = float(signal.confidence)
            else:
                insert_data = {
                    "timestamp": ts_normalized,
                    "market_id": signal.market_id,
                    "strategy_name": signal.strategy_name,
                    "run_id": run_id_val,
                    "prediction_output": signal.prediction_output,
                    "predicted_signal": signal.signal_type.value,
                    "confidence": float(signal.confidence),
                }

                log_entry = self.model_class(**insert_data)
                self.db.add(log_entry)

            if self.commit:
                self.db.commit()
        except Exception as e:
            if self.commit:
                self.db.rollback()
            logger.error(
                f"Failed to log model prediction to database: {e}", exc_info=True
            )
            raise e
