# src/trading_bot/monitoring/prediction_logger.py
import logging
from typing import Sequence

from sqlalchemy.orm import Session

from ..core.schemas import TradeSignal
from ..runs import RunRecorder

logger = logging.getLogger(__name__)


class DatabasePredictionLogger:
    """
    Saves prediction signals for one run through its ``RunRecorder``.
    """

    def __init__(self, db: Session, recorder: RunRecorder, commit: bool = True) -> None:
        """
        :param db: The SQLAlchemy session the run is written to (the replay
                   loop rebinds it to the loop's session).
        :param recorder: The run's recorder (its kind picks live or backtest rows).
        :param commit: If True, commits after each batch; replays commit once at the end.
        """
        self.db = db
        self.recorder = recorder
        self.commit = commit

    @property
    def run_id(self) -> str:
        return self.recorder.run_id

    def on_prediction(self, signal: TradeSignal) -> None:
        """Logs a single TradeSignal."""
        self.log_predictions([signal])

    def log_predictions(self, signals: Sequence[TradeSignal]) -> None:
        """Logs a batch of TradeSignals (upsert per bar, instrument, strategy, run)."""
        if not signals:
            return
        try:
            self.recorder.record_predictions(self.db, signals)
            if self.commit:
                self.db.commit()
        except Exception as e:
            if self.commit:
                self.db.rollback()
            logger.error(
                f"Failed to log prediction batch to database: {e}", exc_info=True
            )
            raise
