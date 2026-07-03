# src/trading_bot/monitoring/prediction_logger.py
import logging
from datetime import timezone
from typing import Any, Optional, Sequence, Type

from sqlalchemy.orm import Session

from ..core.models import ModelPredictionLog
from ..core.schemas import TradeSignal

logger = logging.getLogger(__name__)


class DatabasePredictionLogger:
    """
    Logs prediction signals into a database session in batches.
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
        Logs a single TradeSignal to the database (for backward compatibility).
        """
        self.log_predictions([signal])

    def log_predictions(self, signals: Sequence[TradeSignal]) -> None:
        """
        Logs a batch of TradeSignals to the database using an efficient upsert strategy.
        """
        if not signals:
            return

        run_id_val = self.run_id if self.run_id is not None else "live"
        insert_data = []

        # Determine database dialect
        dialect_name = (
            self.db.bind.dialect.name if self.db and self.db.bind else "sqlite"
        )

        # Retrieve polymorphic discriminator identity for Single Table Inheritance
        polymorphic_identity = getattr(
            self.model_class.__mapper__, "polymorphic_identity", "base"
        )

        for signal in signals:
            # We only log signals that have prediction_output set
            if signal.prediction_output is None:
                continue

            ts = signal.timestamp
            ts_normalized = (
                ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts
            )

            insert_data.append(
                {
                    "timestamp": ts_normalized,
                    "market_id": signal.market_id,
                    "strategy_name": signal.strategy_name,
                    "run_id": run_id_val,
                    "prediction_output": signal.prediction_output,
                    "predicted_signal": signal.signal_type.value,
                    "confidence": float(signal.confidence),
                    "log_type": polymorphic_identity,
                }
            )

        if not insert_data:
            return

        try:
            if dialect_name == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                for row in insert_data:
                    stmt = sqlite_insert(self.model_class).values(row)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[
                            "timestamp",
                            "market_id",
                            "strategy_name",
                            "run_id",
                            "log_type",
                        ],
                        set_={
                            "prediction_output": stmt.excluded.prediction_output,
                            "predicted_signal": stmt.excluded.predicted_signal,
                            "confidence": stmt.excluded.confidence,
                        },
                    )
                    self.db.execute(stmt)
            elif dialect_name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                for row in insert_data:
                    stmt = pg_insert(self.model_class).values(row)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[
                            "timestamp",
                            "market_id",
                            "strategy_name",
                            "run_id",
                            "log_type",
                        ],
                        set_={
                            "prediction_output": stmt.excluded.prediction_output,
                            "predicted_signal": stmt.excluded.predicted_signal,
                            "confidence": stmt.excluded.confidence,
                        },
                    )
                    self.db.execute(stmt)
            else:
                # Fallback to generic bulk insert
                self.db.bulk_insert_mappings(self.model_class, insert_data)

            if self.commit:
                self.db.commit()
        except Exception as e:
            if self.commit:
                self.db.rollback()
            logger.error(
                f"Failed to log prediction batch to database: {e}", exc_info=True
            )
            raise e
