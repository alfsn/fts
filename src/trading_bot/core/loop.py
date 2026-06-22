# src/trading_bot/core/loop.py

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from sqlalchemy.orm import Session

from ..backtesting.abc import BaseBacktestDataReader
from .models import BacktestPredictionLog, ModelPredictionLog
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

    @property
    @abstractmethod
    def prediction_log_model(self) -> Type[Any]:
        """
        Returns the SQLAlchemy ORM model class to use for logging predictions.
        """
        pass

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

    @property
    def prediction_log_model(self) -> Type[Any]:
        return ModelPredictionLog

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

    @property
    def prediction_log_model(self) -> Type[Any]:
        return BacktestPredictionLog

    def __init__(
        self,
        data_reader: Optional[BaseBacktestDataReader] = None,
        data_path: Optional[str] = None,
        save_backtest_report: bool = False,
        backtest_report_dir: Optional[str] = None,
    ) -> None:
        """
        Initializes the backtest replay scheduler.

        :param data_reader: Pluggable backtesting data reader that streams chronological ticks.
        :param data_path: Deprecated data path parameter.
        :param save_backtest_report: Whether to automatically save backtest visualizations as HTML.
        :param backtest_report_dir: Directory where the exported HTML report will be saved.
        """
        self.data_reader = data_reader
        self.data_path = data_path
        self.save_backtest_report = save_backtest_report
        self.backtest_report_dir = backtest_report_dir

    def start(self, pipeline: TradingPipeline, db: Optional[Session] = None) -> None:
        logger.info("Starting HistoricalReplayLoop simulation...")

        reader = self.data_reader
        if not reader and self.data_path:
            # Fallback for backwards compatibility with legacy configurations and test suites
            from ..backtesting.readers import CSVBacktestDataReader

            reader = CSVBacktestDataReader(self.data_path, "mock-btc")

        if not reader:
            logger.warning(
                "No data_reader or data_path provided to HistoricalReplayLoop. Skipping playback."
            )
            return

        # Playback each historical tick sequentially, preventing live execution/ingestion queries
        tick_count = 0
        try:
            for tick_data in reader.read_data():
                logger.debug(
                    f"Replaying historical tick {tick_count + 1} at {tick_data.timestamp}..."
                )
                pipeline.execute_single_tick(db, ingestion_output=tick_data)
                tick_count += 1
        except Exception as e:
            logger.error(
                f"Error during playback at tick {tick_count + 1}: {e}", exc_info=True
            )
            raise e

        logger.info(f"HistoricalReplayLoop complete. Replayed {tick_count} ticks.")

        if self.save_backtest_report:
            logger.info("Automatically saving backtest visualization reports...")
            try:
                from ..backtesting.exporter import HTMLBacktestExporter
                from ..config import settings

                exporter = HTMLBacktestExporter(db_url=settings.DATABASE_URL)
                run_id = (
                    pipeline.prediction_logger.run_id
                    if pipeline.prediction_logger
                    else "All"
                )

                for market_id in pipeline.ingestion.market_ids:
                    for strategy in pipeline.strategy.strategies:
                        try:
                            exporter.export(
                                market_id=market_id,
                                strategy_name=strategy.name,
                                run_id=run_id,
                                output_path=self.backtest_report_dir,
                            )
                        except Exception as export_err:
                            logger.error(
                                f"Failed to automatically export report for market={market_id}, "
                                f"strategy={strategy.name}: {export_err}",
                                exc_info=True,
                            )
            except Exception as err:
                logger.error(
                    f"Failed to initialize backtest visualization exporter: {err}",
                    exc_info=True,
                )
