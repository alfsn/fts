# src/trading_bot/backtesting/simulator.py

import logging
from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.models import BarDataLog
from ..core.schemas import (
    BarData,
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
    OrderBook,
)
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

    def run(self) -> None:
        """Runs the simulation."""
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")

        # 1. Fetch all bars for the given markets and range
        # We sort by timestamp to replay them in order
        stmt = (
            select(BarDataLog)
            .where(BarDataLog.market_id.in_(self.market_ids))
            .where(BarDataLog.timestamp >= self.start_date)
            .where(BarDataLog.timestamp <= self.end_date)
            .order_by(BarDataLog.timestamp.asc())
        )
        bars = self.db.execute(stmt).scalars().all()

        if not bars:
            logger.warning("No bars found for the given criteria.")
            return

        logger.info(f"Replaying {len(bars)} bars.")

        # 2. Replay loop
        for bar_log in bars:
            # Construct a minimal IngestionEngineOutput for the StrategyEngine
            # In backtesting, we wrap the historical bar as the 'latest' data

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

            # Mock MarketData for this tick
            market_data = MarketData(
                market_id=bar_log.market_id,
                order_book=OrderBook(bids=[], asks=[]),  # Empty in simple bar backtest
                recent_trades=[],
                details=MarketDetails(
                    market_id=bar_log.market_id,
                    name=bar_log.market.name,
                    end_date=bar_log.market.end_date,
                    resolution_source=bar_log.market.resolution_source,
                ),
                recent_bars=[bar_data],
            )

            tick_data = IngestionEngineOutput(
                timestamp=bar_log.timestamp,
                market_data={bar_log.market_id: market_data},
                external_data=[],
                bars={bar_log.market_id: [bar_data]},
            )

            # 3. Process the tick
            signals = self.strategy_engine.process_data_tick(tick_data)

            if signals:
                logger.debug(
                    f"Tick {bar_log.timestamp}: Generated {len(signals)} signals."
                )
                # V2: Pass signals to a MockRiskManager and ExecutionSimulator

        logger.info("Backtest simulation complete.")
