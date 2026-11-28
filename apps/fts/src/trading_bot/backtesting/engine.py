# src/trading_bot/backtesting/engine.py

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..backtesting.abc import BaseBacktestDataReader
from ..core.loop import HistoricalReplayLoop
from ..core.pipeline import TradingPipeline
from ..runs import (
    EquityPoint,
    RunHeader,
    Segment,
    SqlRunRecorder,
    check_fills_match,
    delete_run,
    load_record,
)
from ..utils.run_id import generate_backtest_run_id
from .results import BacktestResult

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Coordinates and drives a historical backtesting simulation.

    Integrates with the `HistoricalReplayLoop` to step through historical ticks,
    coordinates data flow through the `TradingPipeline`, records the portfolio
    equity state over time, and generates a structured `BacktestResult`.
    """

    def __init__(
        self,
        pipeline: TradingPipeline,
        data_reader: BaseBacktestDataReader,
        db: Session,
        instrument_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        """
        Initializes the Backtest Engine.

        :param pipeline: The fully wired trading pipeline (strategy, risk, execution).
        :param data_reader: The historical data provider streaming chronological ticks.
        :param db: The one session every row of the run is written to (D16).
        :param instrument_id: Optional market identifier. If omitted, it will be inferred
                          from the data reader if possible.
        :param model_id: The model that trades the run (saved as its segment).
        """
        self.pipeline = pipeline
        self.data_reader = data_reader
        self.db = db
        self.instrument_id = instrument_id or getattr(
            data_reader, "instrument_id", None
        )
        self.model_id = model_id

    def run(
        self, run_id: Optional[str] = None, clear_previous_run: bool = False
    ) -> BacktestResult:
        """
        Executes the backtest simulation run.

        - Writes every row of the run (card, segment, orders, fills,
          predictions, equity) to the session given at construction.
        - Deletes the previous rows of this run_id first if requested.
        - Checks that the saved fills replay to the Portfolio's final state.

        :param run_id: Unique identifier for this backtest run. Auto-generated if omitted or 'backtest_run'.
        :param clear_previous_run: If True, deletes every row of run_id before starting.
        :return: Compiled BacktestResult object.
        """
        if not run_id or run_id == "backtest_run":
            dataset_sha = getattr(self.data_reader, "dataset_sha", None)
            if self.model_id and dataset_sha:
                run_id = generate_backtest_run_id(self.model_id, dataset_sha)
            else:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                short_uid = uuid.uuid4().hex[:8]
                run_id = f"bt_adhoc_{ts}_{short_uid}"

        if clear_previous_run:
            logger.info(f"Clearing previous database rows for run_id: {run_id}")
            delete_run(self.db, run_id)
            self.db.commit()

        portfolio = self.pipeline.portfolio
        strategy_name = (
            self.pipeline.strategy.strategies[0].name
            if (self.pipeline.strategy and self.pipeline.strategy.strategies)
            else "Unknown"
        )
        recorder = SqlRunRecorder(run_id=run_id, kind="backtest")
        recorder.start(
            self.db,
            RunHeader(
                run_id=run_id,
                kind="backtest",
                strategy_name=strategy_name,
                instrument_id=self.instrument_id,
                initial_cash=portfolio.cash_balance,
            ),
        )

        # Orders, fills and equity are committed as they happen, so a run that
        # crashes or fails the fills check is deleted rather than left behind
        # as a half-finished or wrong record.
        try:
            # Every component writes this run through the same recorder and session
            if getattr(self.pipeline, "prediction_logger", None):
                self.pipeline.prediction_logger.recorder = recorder
                self.pipeline.prediction_logger.db = self.db
                self.pipeline.prediction_logger.commit = False
            if getattr(self.pipeline, "execution", None):
                self.pipeline.execution.recorder = recorder

            equity_curve: List[Dict[str, Any]] = []

            def on_tick_callback(tick_data: Any) -> None:
                # Equity comes from the portfolio's single valuation (PR 7b):
                # every position at its last close, carried over missing bars.
                market_data_map = {m.instrument_id: m for m in tick_data.market_data}
                equity = portfolio.equity(market_data_map)

                pos_size = 0.0
                close_price = 0.0  # no single traded instrument: no close column
                if self.instrument_id:
                    close = portfolio.last_close(self.instrument_id)
                    if close is None:
                        return  # no price seen yet: nothing to value
                    close_price = close
                    pos = portfolio.positions.get(self.instrument_id)
                    if pos:
                        pos_size = pos.quantity

                point = EquityPoint(
                    timestamp=tick_data.timestamp,
                    cash=portfolio.cash_balance,
                    position=pos_size,
                    close=close_price,
                    equity=equity,
                )
                equity_curve.append(point.model_dump())
                recorder.record_equity(self.db, point)

            logger.info(f"Starting backtest engine simulation for run_id: {run_id}")
            loop_driver = HistoricalReplayLoop(data_reader=self.data_reader)
            loop_driver.start(self.pipeline, db=self.db, on_tick=on_tick_callback)

            start = equity_curve[0]["timestamp"] if equity_curve else None
            end = equity_curve[-1]["timestamp"] if equity_curve else None
            recorder.add_segment(
                self.db, Segment(model_id=self.model_id, start_time=start, end_time=end)
            )
            recorder.finish(self.db, start, end)
            self.db.commit()

            # Fills are the truth (D19): they must replay to the final portfolio.
            record = load_record(self.db, run_id, predictions=False)
            check_fills_match(
                record,
                positions={k: p.quantity for k, p in portfolio.positions.items()},
                cash=portfolio.cash_balance,
                currency=portfolio.quote_currency,
            )
        except Exception:
            logger.exception(f"Run {run_id} did not finish; deleting its rows.")
            self.db.rollback()
            delete_run(self.db, run_id)
            self.db.commit()
            raise

        return BacktestResult(
            run_id=run_id,
            instrument_id=self.instrument_id or "Unknown",
            strategy_name=strategy_name,
            record=record,
        )
