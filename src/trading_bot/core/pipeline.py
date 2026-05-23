# src/trading_bot/core/pipeline.py

import logging
from typing import Optional

from sqlalchemy.orm import Session

from ..data_ingestion.engine import DataIngestionEngine
from ..execution.engine import ExecutionEngine
from ..risk_management.manager import RiskManager
from ..risk_management.portfolio import Portfolio
from ..strategy.engine import StrategyEngine

logger = logging.getLogger(__name__)


class TradingPipeline:
    """
    Coordinates and drives a single tick of the trading lifecycle.

    Adheres to the Single Responsibility Principle by only managing the
    data flow and step-by-step sequence of a tick (Ingestion -> Strategy -> Risk -> Execution).
    It has no knowledge of timing, scheduling, or loop models, which are delegated to loop drivers.
    """

    def __init__(
        self,
        ingestion: DataIngestionEngine,
        strategy: StrategyEngine,
        risk: RiskManager,
        execution: ExecutionEngine,
        portfolio: Portfolio,
    ) -> None:
        """
        Initializes the pipeline with injected core components.
        """
        self.ingestion = ingestion
        self.strategy = strategy
        self.risk = risk
        self.execution = execution
        self.portfolio = portfolio

    def execute_single_tick(self, db: Optional[Session] = None) -> None:
        """
        Executes one complete pass of the trading bot pipeline.

        1. Fetches all market and external data.
        2. Feeds data to the Strategy Engine to generate raw trading signals.
        3. Passes each signal to the Risk Manager for validation and sizing.
        4. Routes approved orders to the Execution Engine for fill placement.

        :param db: The optional SQLAlchemy database session.
        """
        logger.debug("--- Starting pipeline tick ---")
        try:
            # Step 1: Ingest Data
            ingestion_output = self.ingestion.fetch_all_data()
            if not ingestion_output.market_data:
                logger.debug("No active market data received this tick.")
                return

            # Step 2: Generate Strategy Signals
            signals = self.strategy.process_data_tick(ingestion_output)
            if not signals:
                logger.debug("No trading signals generated this tick.")
                return

            # Step 3: Process Signals through Risk & Route for Execution
            for signal in signals:
                market_data = ingestion_output.market_data.get(signal.market_id)
                if not market_data:
                    logger.warning(
                        f"Skipping signal for {signal.market_id}: no market data "
                        f"available in ingestion output."
                    )
                    continue

                # Run risk analysis and position sizing
                order_request = self.risk.process_signal(
                    signal=signal,
                    market_data=market_data,
                    db=db,
                )

                if order_request:
                    logger.info(
                        f"Risk approved order for {signal.market_id} "
                        f"(size: {order_request.size:.4f} shares)."
                    )
                    # Step 4: Execute Order
                    self.execution.execute_order(
                        order=order_request,
                        db=db,
                        strategy_name=signal.strategy_name,
                    )
                else:
                    logger.debug(
                        f"Signal for {signal.market_id} rejected by risk manager."
                    )

        except Exception as e:
            logger.error(
                f"Critical error during pipeline execution: {e}", exc_info=True
            )
        logger.debug("--- Finished pipeline tick ---")
