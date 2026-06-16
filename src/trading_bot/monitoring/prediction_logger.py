import logging

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

    def __init__(self, db: Session, commit: bool = True) -> None:
        """
        Initializes the database prediction logger.

        :param db: The SQLAlchemy DB Session.
        :param commit: If True, commits the transaction immediately.
        """
        self.db = db
        self.commit = commit

    def on_prediction(self, signal: TradeSignal) -> None:
        """
        Logs a generated TradeSignal to the database.
        """
        try:
            log_entry = ModelPredictionLog(
                timestamp=signal.timestamp,
                market_id=signal.market_id,
                strategy_name=signal.strategy_name,
                prediction_output=signal.prediction_output,
                predicted_signal=signal.signal_type.value,
                confidence=float(signal.confidence),
            )
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
