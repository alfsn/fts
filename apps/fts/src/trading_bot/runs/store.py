# src/trading_bot/runs/store.py
"""Read and delete side of runs (M1): the only reader of the run tables."""

from typing import Callable, Dict, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .records import (
    EquityPoint,
    Fill,
    Prediction,
    RunHeader,
    RunKind,
    RunRecord,
    Segment,
)
from .tables import (
    RUN_TABLES,
    BacktestEquityLog,
    BacktestPredictionLog,
    OrderLog,
    PredictionLog,
    RunLog,
    RunSegmentLog,
    TradeLog,
)


def _header(run: RunLog) -> RunHeader:
    return RunHeader(
        run_id=run.run_id,
        kind=run.kind,
        strategy_name=run.strategy_name,
        instrument_id=run.instrument_id,
        initial_cash=run.initial_cash,
        start_time=run.start_time,
        end_time=run.end_time,
    )


def load_record(
    session: Session, run_id: str, predictions: bool = True
) -> Optional[RunRecord]:
    """One run from ``session`` (sees its uncommitted rows), or None."""
    run = session.get(RunLog, run_id)
    if run is None:
        return None
    segments = session.scalars(
        select(RunSegmentLog)
        .where(RunSegmentLog.run_id == run_id)
        .order_by(RunSegmentLog.id)
    ).all()
    equity = session.scalars(
        select(BacktestEquityLog)
        .where(BacktestEquityLog.run_id == run_id)
        .order_by(BacktestEquityLog.timestamp)
    ).all()
    fills = session.scalars(
        select(TradeLog)
        .where(TradeLog.run_id == run_id)
        .order_by(TradeLog.fill_timestamp, TradeLog.id)
    ).all()
    preds = (
        session.scalars(
            select(PredictionLog)
            .where(PredictionLog.run_id == run_id)
            .order_by(PredictionLog.timestamp)
        ).all()
        if predictions
        else []
    )
    return RunRecord(
        header=_header(run),
        segments=[
            Segment(model_id=s.model_id, start_time=s.start_time, end_time=s.end_time)
            for s in segments
        ],
        equity=[
            EquityPoint(
                timestamp=e.timestamp,
                cash=e.cash,
                position=e.position,
                close=e.close,
                equity=e.equity,
            )
            for e in equity
        ],
        fills=[
            Fill(
                order_id=t.order_id,
                instrument_id=t.instrument_id,
                side=t.side,
                outcome=t.outcome,
                quantity=t.quantity,
                fill_price=t.fill_price,
                fill_timestamp=t.fill_timestamp,
            )
            for t in fills
        ],
        predictions=[
            Prediction(
                timestamp=p.timestamp,
                instrument_id=p.instrument_id,
                strategy_name=p.strategy_name,
                prediction_output=p.prediction_output,
                predicted_signal=p.predicted_signal,
                confidence=p.confidence,
                actual_future_return=p.actual_future_return,
            )
            for p in preds
        ],
    )


def delete_run(session: Session, run_id: str) -> None:
    """Remove every row of the run from every run table (no commit)."""
    for table in RUN_TABLES:
        session.execute(delete(table).where(table.c.run_id == run_id))


class RunStore:
    """Lists, loads and deletes whole runs."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    def list(
        self, kind: RunKind = "backtest", instrument_id: Optional[str] = None
    ) -> List[RunHeader]:
        """Run cards, newest first."""
        stmt = select(RunLog).where(RunLog.kind == kind)
        if instrument_id:
            stmt = stmt.where(RunLog.instrument_id == instrument_id)
        with self.session_factory() as session:
            runs = session.scalars(stmt.order_by(RunLog.created_at.desc())).all()
            return [_header(r) for r in runs]

    def load(self, run_id: str, predictions: bool = True) -> Optional[RunRecord]:
        with self.session_factory() as session:
            return load_record(session, run_id, predictions=predictions)

    def delete(self, run_id: str) -> None:
        with self.session_factory() as session:
            delete_run(session, run_id)
            session.commit()

    def counts(self) -> Dict[str, int]:
        """Row counts for the dashboard's database summary."""

        def count(session: Session, table, *where) -> int:
            stmt = select(func.count()).select_from(table).where(*where)
            return session.scalar(stmt) or 0

        with self.session_factory() as session:
            return {
                "runs": count(session, RunLog),
                "order_logs": count(session, OrderLog),
                "trade_logs": count(session, TradeLog),
                "equity_log_ticks": count(session, BacktestEquityLog),
                "prediction_logs": count(
                    session,
                    PredictionLog,
                    PredictionLog.log_type
                    == BacktestPredictionLog.__mapper__.polymorphic_identity,
                ),
            }
