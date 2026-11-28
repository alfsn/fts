# src/trading_bot/runs/recorder.py
"""
Write side of a run (M1): what pipeline components call while a run happens.

Every method writes into the session it is given (D16: one session per run)
and never commits; the loop or the execution engine owns the transaction.
"""

from datetime import datetime, timezone
from typing import Optional, Protocol, Sequence

from sqlalchemy.orm import Session

from ..core.enums import OrderSide, OrderStatus
from ..core.schemas import ExecutionResult, OrderRequest, TradeSignal
from .records import EquityPoint, RunHeader, RunKind, Segment
from .tables import (
    BacktestEquityLog,
    BacktestPredictionLog,
    ModelPredictionLog,
    OrderLog,
    PredictionLog,
    RunLog,
    RunSegmentLog,
    TradeLog,
)

_PREDICTION_TABLE = {"backtest": BacktestPredictionLog, "live": ModelPredictionLog}


class RunRecorder(Protocol):
    run_id: str
    kind: RunKind

    def start(self, db: Session, header: RunHeader) -> None: ...

    def finish(
        self, db: Session, start_time: Optional[datetime], end_time: Optional[datetime]
    ) -> None: ...

    def add_segment(self, db: Session, segment: Segment) -> None: ...

    def record_order(
        self,
        db: Session,
        order: OrderRequest,
        result: ExecutionResult,
        strategy_name: str,
    ) -> None: ...

    def update_order(self, db: Session, result: ExecutionResult) -> bool: ...

    def order_status(self, db: Session, order_id: str) -> Optional[OrderStatus]: ...

    def record_unknown_order(self, db: Session, result: ExecutionResult) -> None: ...

    def record_fill(
        self, db: Session, order: OrderRequest, result: ExecutionResult
    ) -> None: ...

    def record_equity(self, db: Session, point: EquityPoint) -> None: ...

    def record_predictions(
        self, db: Session, signals: Sequence[TradeSignal]
    ) -> None: ...


class RunExistsError(ValueError):
    """A run with this run_id is already saved."""


class SqlRunRecorder:
    """Writes one run's rows, all stamped with ``run_id``."""

    def __init__(self, run_id: str, kind: RunKind) -> None:
        if not run_id:
            raise ValueError("A run needs a run_id.")
        self.run_id = run_id
        self.kind = kind

    # --- the run card ----------------------------------------------------------

    def start(self, db: Session, header: RunHeader) -> None:
        """Write the run card. A run_id is used once: starting it again would
        add a second run's orders, fills and equity to the first one's."""
        if header.run_id != self.run_id or header.kind != self.kind:
            raise ValueError("The header belongs to another run.")
        if db.get(RunLog, self.run_id) is not None:
            raise RunExistsError(
                f"Run {self.run_id} already exists. Delete it first "
                "(RunStore.delete, or BacktestEngine.run(clear_previous_run=True))."
            )
        db.add(RunLog(**header.model_dump()))

    def finish(
        self, db: Session, start_time: Optional[datetime], end_time: Optional[datetime]
    ) -> None:
        run = db.get(RunLog, self.run_id)
        if run is None:
            raise ValueError(f"Run {self.run_id} was never started.")
        run.start_time, run.end_time = start_time, end_time

    def add_segment(self, db: Session, segment: Segment) -> None:
        db.add(RunSegmentLog(run_id=self.run_id, **segment.model_dump()))

    # --- orders and fills -----------------------------------------------------

    def _order(self, db: Session, order_id: str) -> Optional[OrderLog]:
        return (
            db.query(OrderLog).filter_by(run_id=self.run_id, order_id=order_id).first()
        )

    @staticmethod
    def _apply(log: OrderLog, result: ExecutionResult) -> None:
        log.status = result.status
        log.filled_quantity = result.filled_quantity
        log.avg_fill_price = result.avg_price
        log.updated_at = datetime.now(timezone.utc)

    def record_order(
        self,
        db: Session,
        order: OrderRequest,
        result: ExecutionResult,
        strategy_name: str,
    ) -> None:
        log = self._order(db, result.order_id)
        if log is None:
            log = OrderLog(
                order_id=result.order_id,
                run_id=self.run_id,
                instrument_id=order.instrument_id,
                strategy_name=strategy_name,
                side=order.side,
                outcome=order.outcome,
                requested_quantity=order.quantity,
                requested_price=order.price,
            )
            db.add(log)
        self._apply(log, result)

    def update_order(self, db: Session, result: ExecutionResult) -> bool:
        log = self._order(db, result.order_id)
        if log is None:
            return False
        self._apply(log, result)
        return True

    def order_status(self, db: Session, order_id: str) -> Optional[OrderStatus]:
        log = self._order(db, order_id)
        return None if log is None else log.status

    def record_unknown_order(self, db: Session, result: ExecutionResult) -> None:
        """An order the venue knows but we never logged (best-effort recovery)."""
        db.add(
            OrderLog(
                order_id=result.order_id,
                run_id=self.run_id,
                instrument_id="UNKNOWN_MARKET",
                strategy_name="reconciled",
                side=OrderSide.BUY,
                requested_quantity=result.filled_quantity,
                requested_price=result.avg_price,
                status=result.status,
                filled_quantity=result.filled_quantity,
                avg_fill_price=result.avg_price,
            )
        )

    def record_fill(
        self, db: Session, order: OrderRequest, result: ExecutionResult
    ) -> None:
        """One row per fill the Portfolio applied, at the fill's bar time.

        Idempotent per order: the Portfolio fills an order once, so a retried
        save (reconciliation after a failed commit) never adds a second row."""
        exists = (
            db.query(TradeLog.id)
            .filter_by(run_id=self.run_id, order_id=result.order_id)
            .first()
        )
        if exists is not None:
            return
        db.add(
            TradeLog(
                run_id=self.run_id,
                order_id=result.order_id,
                instrument_id=order.instrument_id,
                side=order.side,
                outcome=order.outcome,
                quantity=result.filled_quantity,
                fill_price=result.avg_price,
                fill_timestamp=result.timestamp,
            )
        )

    # --- equity and predictions -----------------------------------------------

    def record_equity(self, db: Session, point: EquityPoint) -> None:
        db.add(BacktestEquityLog(run_id=self.run_id, **point.model_dump()))

    def record_predictions(self, db: Session, signals: Sequence[TradeSignal]) -> None:
        """Upsert the signals that carry a model output (one row per bar,
        instrument, strategy and run)."""
        table = _PREDICTION_TABLE[self.kind]
        log_type = table.__mapper__.polymorphic_identity
        rows = []
        for signal in signals:
            if signal.prediction_output is None:
                continue
            ts = signal.timestamp
            rows.append(
                {
                    "timestamp": (
                        ts.astimezone(timezone.utc).replace(tzinfo=None)
                        if ts.tzinfo
                        else ts
                    ),
                    "instrument_id": signal.instrument_id,
                    "strategy_name": signal.strategy_name,
                    "run_id": self.run_id,
                    "prediction_output": signal.prediction_output,
                    "predicted_signal": signal.signal_type.value,
                    "confidence": float(signal.confidence),
                    "log_type": log_type,
                }
            )
        if not rows:
            return

        dialect = db.bind.dialect.name if db.bind else "sqlite"
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            db.bulk_insert_mappings(table, rows)
            return

        for row in rows:
            stmt = insert(PredictionLog).values(row)
            db.execute(
                stmt.on_conflict_do_update(
                    index_elements=[
                        "timestamp",
                        "instrument_id",
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
            )
