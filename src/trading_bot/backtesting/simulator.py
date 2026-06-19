# src/trading_bot/backtesting/simulator.py

import hashlib
import logging
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy.orm import Session

from ..core.models import BacktestPredictionLog
from ..core.models import BarDataLog as BarDataLogModel
from ..core.repository import MarketDataRepository
from ..core.schemas import (
    BarData,
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
    OrderBook,
)
from ..monitoring.prediction_logger import DatabasePredictionLogger
from ..strategy.engine import StrategyEngine

logger = logging.getLogger(__name__)


class BacktestSimulator:
    """
    Simulates live trading by replaying historical bars from the database.
    """

    def __init__(
        self,
        db: Session,
        strategy_engine: StrategyEngine,
        market_ids: Sequence[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        self.db = db
        self.strategy_engine = strategy_engine
        self.market_ids = market_ids
        self.start_date = start_date
        self.end_date = end_date
        self.repository = MarketDataRepository(db)

    def run(self) -> None:
        """Runs the simulation."""

        # Generate a unique backtest run ID using SHA-256 hash of current timestamp
        start_ts = datetime.now(timezone.utc).isoformat()
        backtest_id = hashlib.sha256(start_ts.encode("utf-8")).hexdigest()[:16]

        logger.info(
            f"Starting backtest (id: {backtest_id}) from {self.start_date} to {self.end_date}"
        )

        backtest_logger = DatabasePredictionLogger(
            db=self.db,
            commit=True,
            model_class=BacktestPredictionLog,
            backtest_id=backtest_id,
        )

        # Register the observer dynamically to any strategy that supports observers
        registered_observers = []
        for strategy in self.strategy_engine.strategies:
            if hasattr(strategy, "observers") and isinstance(strategy.observers, list):
                strategy.observers.append(backtest_logger)
                registered_observers.append((strategy, backtest_logger))

        try:
            # 1. Fetch all bars for the given markets and range using repository
            bars = self.repository.get_bars(
                market_ids=self.market_ids,
                start_date=self.start_date,
                end_date=self.end_date,
            )

            if not bars:
                logger.warning("No bars found for the given criteria.")
                return

            logger.info(f"Replaying {len(bars)} bars.")

            # Pre-populate historical bars for lookback from the database before start_date
            historical_bars = {}
            for market_id in self.market_ids:
                prior_bar_logs = (
                    self.db.query(BarDataLogModel)
                    .filter(
                        BarDataLogModel.market_id == market_id,
                        BarDataLogModel.timestamp < self.start_date,
                    )
                    .order_by(BarDataLogModel.timestamp.desc())
                    .limit(100)
                    .all()
                )
                prior_bar_logs.reverse()

                historical_bars[market_id] = [
                    BarData(
                        timestamp=b.timestamp,
                        open=b.open,
                        high=b.high,
                        low=b.low,
                        close=b.close,
                        volume=b.volume,
                        bar_type=b.bar_type,
                        ticks_count=b.ticks_count,
                        dollar_volume=b.dollar_volume,
                    )
                    for b in prior_bar_logs
                ]

            # 2. Replay loop
            for bar_log in bars:
                # Convert DB model to Pydantic BarData
                bar_data = BarData(
                    timestamp=bar_log.timestamp,
                    open=bar_log.open,
                    high=bar_log.high,
                    low=bar_log.low,
                    close=bar_log.close,
                    volume=bar_log.volume,
                    bar_type=bar_log.bar_type,
                    ticks_count=bar_log.ticks_count,
                    dollar_volume=bar_log.dollar_volume,
                )

                if bar_log.market_id not in historical_bars:
                    historical_bars[bar_log.market_id] = []
                historical_bars[bar_log.market_id].append(bar_data)

                # Slide window to avoid keeping unbounded history in memory
                if len(historical_bars[bar_log.market_id]) > 1000:
                    historical_bars[bar_log.market_id].pop(0)

                # Mock MarketData for this tick (with empty order book as requested)
                market_data = MarketData(
                    market_id=bar_log.market_id,
                    order_book=OrderBook(
                        bids=[], asks=[]
                    ),  # Empty in simple bar backtest
                    recent_trades=[],
                    details=MarketDetails(
                        market_id=bar_log.market_id,
                        name=bar_log.market.name,
                        end_date=bar_log.market.end_date,
                        resolution_source=bar_log.market.resolution_source,
                    ),
                    recent_bars=historical_bars[bar_log.market_id],
                )

                tick_data = IngestionEngineOutput(
                    timestamp=bar_log.timestamp,
                    market_data={bar_log.market_id: market_data},
                    external_data=[],
                    bars={bar_log.market_id: historical_bars[bar_log.market_id]},
                )

                # 3. Process the tick
                signals = self.strategy_engine.process_data_tick(tick_data)

                if signals:
                    logger.debug(
                        f"Tick {bar_log.timestamp}: Generated {len(signals)} signals."
                    )
                    # V2: Pass signals to a MockRiskManager and ExecutionSimulator

            logger.info("Backtest simulation complete.")
        finally:
            # Clean up the registered observers to prevent pollution/leaks
            for strategy, obs in registered_observers:
                if obs in strategy.observers:
                    strategy.observers.remove(obs)
