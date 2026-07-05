# src/trading_bot/backtesting/engine.py

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..backtesting.abc import BaseBacktestDataReader
from ..config import settings
from ..core.database import Base
from ..core.loop import HistoricalReplayLoop
from ..core.models import (
    BacktestEquityLog,
    BacktestPredictionLog,
    OrderLog,
    Position,
    TradeLog,
)
from ..core.pipeline import TradingPipeline
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
        market_id: Optional[str] = None,
    ):
        """
        Initializes the Backtest Engine.

        :param pipeline: The fully wired trading pipeline (strategy, risk, execution).
        :param data_reader: The historical data provider streaming chronological ticks.
        :param db: SQLAlchemy database session for order persistence.
        :param market_id: Optional market identifier. If omitted, it will be inferred
                          from the data reader if possible.
        """
        self.pipeline = pipeline
        self.data_reader = data_reader
        self.db = db
        self.market_id = market_id or getattr(data_reader, "market_id", None)

    def run(
        self, run_id: Optional[str] = None, clear_previous_run: bool = False
    ) -> BacktestResult:
        """
        Executes the backtest simulation run.

        - Connects dynamically to an isolated backtests.db physical SQLite database.
        - Purges historical logs for this run_id if requested.
        - Synchronizes run_id across pipeline components.
        - Runs the replay loop while tracking cash, positions, and equity.
        - Returns a detailed BacktestResult.

        :param run_id: Unique identifier for this backtest run. Auto-generated if omitted or 'backtest_run'.
        :param clear_previous_run: If True, deletes database records matching run_id before starting.
        :return: Compiled BacktestResult object.
        """
        if not run_id or run_id == "backtest_run":
            model_id = getattr(self.pipeline, "model_id", None) or getattr(
                self, "model_id", None
            )
            dataset_sha = getattr(self.data_reader, "dataset_sha", None)
            if model_id and dataset_sha:
                run_id = generate_backtest_run_id(model_id, dataset_sha)
            else:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                short_uid = uuid.uuid4().hex[:8]
                run_id = f"bt_adhoc_{ts}_{short_uid}"

        db_url = str(self.db.bind.url) if self.db and self.db.bind else ""
        is_test_env = ":memory:" in db_url or "test" in db_url
        orig_db = self.db

        if not is_test_env:
            from ..core.database import create_db_session, init_db

            backtest_db_url = settings.BACKTEST_DATABASE_URL
            logger.info(
                f"Isolating backtest data by binding database session to: {backtest_db_url}"
            )
            self.db = create_db_session(backtest_db_url)

        try:
            # Self-healing database tables creation
            try:
                if not is_test_env:
                    init_db(bind_engine=self.db.bind)
                else:
                    from ..core.database import init_db

                    init_db(bind_engine=self.db.bind)
            except Exception as err:
                logger.error(
                    f"Failed to auto-initialize isolated backtest database: {err}"
                )

            if clear_previous_run:
                logger.info(f"Clearing previous database logs for run_id: {run_id}")
                try:
                    self.db.query(BacktestPredictionLog).filter_by(
                        run_id=run_id
                    ).delete()
                    self.db.query(OrderLog).filter_by(run_id=run_id).delete()
                    self.db.query(TradeLog).filter_by(run_id=run_id).delete()
                    self.db.query(Position).filter_by(run_id=run_id).delete()
                    self.db.query(BacktestEquityLog).filter_by(run_id=run_id).delete()
                    self.db.commit()
                except Exception as e:
                    self.db.rollback()
                    logger.error(
                        f"Failed to clear database logs for run_id {run_id}: {e}"
                    )
                    raise e

            # Synchronize run_id configuration in pipeline components
            if (
                hasattr(self.pipeline, "prediction_logger")
                and self.pipeline.prediction_logger
            ):
                self.pipeline.prediction_logger.run_id = run_id
                self.pipeline.prediction_logger.db = self.db
                self.pipeline.prediction_logger.commit = False

            if hasattr(self.pipeline, "execution") and self.pipeline.execution:
                self.pipeline.execution.run_id = run_id

            equity_curve: List[Dict[str, Any]] = []

            def on_tick_callback(tick_data: Any) -> None:
                # Track portfolio balances and position size (using public accessors)
                current_cash = self.pipeline.portfolio.cash_balance

                pos_size = 0.0
                close_price = 0.0

                if self.market_id:
                    # Retrieve active position size
                    pos = self.pipeline.portfolio.positions.get(self.market_id)
                    if pos:
                        pos_size = pos.size

                    # Retrieve last close price for equity calculation
                    market_data = tick_data.market_data.get(self.market_id)
                    if market_data and market_data.recent_bars:
                        close_price = market_data.recent_bars[-1].close

                # Equity = Cash + Position Value (valued at current close price)
                equity = current_cash + (pos_size * close_price)

                equity_curve.append(
                    {
                        "timestamp": tick_data.timestamp,
                        "cash": current_cash,
                        "position": pos_size,
                        "close": close_price,
                        "equity": equity,
                    }
                )

                # Persist the actual equity curve point to the database
                equity_log = BacktestEquityLog(
                    run_id=run_id,
                    timestamp=tick_data.timestamp,
                    cash=current_cash,
                    position=pos_size,
                    close=close_price,
                    equity=equity,
                )
                self.db.add(equity_log)

            logger.info(f"Starting backtest engine simulation for run_id: {run_id}")
            loop_driver = HistoricalReplayLoop(data_reader=self.data_reader)
            loop_driver.start(self.pipeline, db=self.db, on_tick=on_tick_callback)

            # Explicitly commit to ensure all logged predictions, orders, and equity logs are persisted
            self.db.commit()

            result = BacktestResult(
                run_id=run_id,
                market_id=self.market_id or "Unknown",
                strategy_name=(
                    self.pipeline.strategy.strategies[0].name
                    if (self.pipeline.strategy and self.pipeline.strategy.strategies)
                    else "Unknown"
                ),
                db_session=self.db,
            )
            return result
        finally:
            if not is_test_env:
                self.db.close()
            self.db = orig_db
