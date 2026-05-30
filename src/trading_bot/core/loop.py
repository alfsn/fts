# src/trading_bot/core/loop.py

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from .pipeline import TradingPipeline
from .schemas import IngestionEngineOutput, MarketData, MarketDetails

logger = logging.getLogger(__name__)


class BaseEventLoop(ABC):
    """
    Abstract Base Class for driving the FTS Trading Pipeline.

    Adheres to the SOLID principles by establishing a common interface
    for running the pipeline under different scheduling models (Real-time polling,
    Websocket subscriptions, or Historical backtest replays).
    """

    @abstractmethod
    def start(self, pipeline: TradingPipeline, db: Optional[Session] = None) -> None:
        """
        Drives the event loop scheduler and runs the pipeline.

        :param pipeline: The instantiated TradingPipeline orchestration.
        :param db: The SQLAlchemy database session.
        """
        pass


class RealTimePollingLoop(BaseEventLoop):
    """
    Drives the pipeline on a real-time periodic schedule (polling).
    Uses a standard blocking time.sleep() timer mechanism.
    """

    def __init__(
        self, interval_seconds: float, max_ticks: Optional[int] = None
    ) -> None:
        """
        Initializes the polling loop.

        :param interval_seconds: The duration of the sleep interval between ticks.
        :param max_ticks: Optional threshold to limit running cycles (useful for testing).
        """
        self.interval = interval_seconds
        self.max_ticks = max_ticks
        self._running = False

    def start(self, pipeline: TradingPipeline, db: Optional[Session] = None) -> None:
        self._running = True
        tick_count = 0

        logger.info(
            f"Starting RealTimePollingLoop with interval={self.interval}s "
            f"(max_ticks={self.max_ticks if self.max_ticks is not None else 'infinite'})."
        )

        try:
            while self._running:
                logger.debug(f"Executing polling tick {tick_count + 1}...")
                start_time = time.time()
                pipeline.execute_single_tick(db)
                tick_count += 1

                if self.max_ticks is not None and tick_count >= self.max_ticks:
                    logger.info(
                        "Max ticks threshold reached. Terminating polling loop."
                    )
                    break

                # Sleep until next scheduled interval, correcting for tick execution time
                elapsed = time.time() - start_time
                sleep_time = max(0.0, self.interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Loop execution interrupted by user (KeyboardInterrupt).")
        finally:
            self._running = False
            logger.info("RealTimePollingLoop halted.")

    def stop(self) -> None:
        """Stops loop execution at the end of the current tick."""
        self._running = False


class HistoricalReplayLoop(BaseEventLoop):
    """
    Drives the pipeline sequentially over historical data ticks.
    Used for local backtesting simulation.
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        """
        Initializes the backtest replay scheduler.

        :param data_path: Path to the file containing historical price series or trade data.
        """
        self.data_path = data_path

    def start(self, pipeline: TradingPipeline, db: Optional[Session] = None) -> None:
        logger.info(f"Starting HistoricalReplayLoop loading from: {self.data_path}")

        # In a real backtesting scenario, we would load the dataset here,
        # set up a simulated broker / virtual order fills, and feed
        # price updates one by one to the ingestion engine before executing a tick.
        #
        # For this version, we provide a structured placeholder that executes a single mock pass safely
        # without invoking live network/database queries from the production IngestionEngine.
        logger.debug(
            "Running historical backtest simulation tick safely using offline mock ingestion..."
        )

        mock_details = MarketDetails(
            market_id="mock-btc",
            name="Will BTC exceed 100k?",
            end_date=datetime.now(timezone.utc),
            resolution_source="oracle",
        )
        mock_market_data = MarketData(
            market_id="mock-btc",
            details=mock_details,
            recent_bars=[],
        )
        mock_ingestion_output = IngestionEngineOutput(
            timestamp=datetime.now(timezone.utc),
            market_data={"mock-btc": mock_market_data},
            external_data=[],
        )

        pipeline.execute_single_tick(db, ingestion_output=mock_ingestion_output)
        logger.info("HistoricalReplayLoop complete.")
