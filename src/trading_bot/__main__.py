# src/trading_bot/__main__.py

import argparse
import logging
import sys
from pathlib import Path

import yaml

from trading_bot.config import PluginLoader, TaskConfig, settings
from trading_bot.core.database import SessionLocal, init_db
from trading_bot.core.loop import BaseEventLoop
from trading_bot.core.pipeline import TradingPipeline
from trading_bot.core.repository import OrderRepository, PositionRepository
from trading_bot.risk_management.portfolio import Portfolio

# Setup logging
logging.basicConfig(
    level=logging.getLevelName(settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("trading_bot.runner")


def main() -> None:
    """
    Main runner entrypoint that dynamically loads tasks from YAML,
    wires S.O.L.I.D. dependencies recursively, and drives execution.
    """
    parser = argparse.ArgumentParser(description="FTS Trading Bot Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="nets_task.yaml",
        help="Path to the task YAML configuration file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(
            f"Configuration file not found: {args.config}. "
            "Please create it or specify a valid file path using --config."
        )
        sys.exit(1)

    logger.info(f"Loading task configuration from: {config_path}")

    # 1. Parse YAML into TaskConfig schema
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        task_config = TaskConfig(**raw_config)
    except Exception as e:
        logger.critical(f"Failed to parse task configuration: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Successfully loaded task: '{task_config.name}'")

    # 2. Initialize Database & Register Plugin Models
    try:
        init_db(extra_models=task_config.extra_models)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # 3. Create DB Session and Repositories
    db_session = SessionLocal()
    pos_repo = PositionRepository(db_session)
    order_repo = OrderRepository(db_session)

    try:
        # 4. Initialize Portfolio State (injecting decoupled repositories)
        portfolio = Portfolio(
            initial_balance=10000.0,
            quote_currency="USD",
            pos_repo=pos_repo,
            order_repo=order_repo,
        )
        portfolio.load_positions()

        # 5. Dynamic Recursive Instantiation via PluginLoader
        logger.info("Initializing dynamic core and plugin components...")
        market_provider = PluginLoader.instantiate(task_config.market_provider)

        external_providers = [
            PluginLoader.instantiate(cfg) for cfg in task_config.external_providers
        ]

        strategies = [PluginLoader.instantiate(cfg) for cfg in task_config.strategies]

        sizing_strategy = PluginLoader.instantiate(task_config.sizing_strategy)

        if task_config.execution_handler:
            execution_handler = PluginLoader.instantiate(task_config.execution_handler)
        else:
            from trading_bot.execution.handlers.polymarket_handler import (
                PolymarketHandler,
            )

            execution_handler = PolymarketHandler()
            logger.info(
                "No execution_handler configured. Defaulting to mock PolymarketHandler."
            )

        # 6. Bind into Core Orchestration Engines
        from trading_bot.data_ingestion.engine import DataIngestionEngine
        from trading_bot.execution.engine import ExecutionEngine
        from trading_bot.risk_management.manager import RiskManager
        from trading_bot.strategy.engine import StrategyEngine

        ingestion_engine = DataIngestionEngine(
            market_provider=market_provider,
            external_providers=external_providers,
            market_ids=task_config.market_ids,
        )

        strategy_engine = StrategyEngine(strategies=strategies)

        risk_manager = RiskManager(
            portfolio=portfolio,
            sizing_strategy=sizing_strategy,
        )

        execution_engine = ExecutionEngine(
            execution_handler=execution_handler,
            portfolio=portfolio,
        )

        # 7. Bind Unified TradingPipeline
        pipeline = TradingPipeline(
            ingestion=ingestion_engine,
            strategy=strategy_engine,
            risk=risk_manager,
            execution=execution_engine,
            portfolio=portfolio,
        )

        # 8. Instantiate and Start Configured Loop Driver
        if not task_config.loop_driver:
            logger.critical("No loop_driver component configured in task YAML!")
            sys.exit(1)

        loop_driver = PluginLoader.instantiate(task_config.loop_driver)
        if not isinstance(loop_driver, BaseEventLoop):
            logger.critical(
                f"Configured loop_driver '{task_config.loop_driver.class_path}' "
                f"must inherit from BaseEventLoop!"
            )
            sys.exit(1)

        logger.info(
            f"Event loop starting via driver: '{task_config.loop_driver.class_path}'"
        )
        loop_driver.start(pipeline, db=db_session)

    except Exception as e:
        logger.critical(f"Runner encountered critical error: {e}", exc_info=True)
    finally:
        logger.info("Closing database connection.")
        db_session.close()
        logger.info("FTS Runner stopped.")


if __name__ == "__main__":
    main()
