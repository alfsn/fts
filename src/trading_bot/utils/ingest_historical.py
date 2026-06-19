# src/trading_bot/utils/ingest_historical.py

import argparse
import logging
import sys

from trading_bot.core.database import SessionLocal, init_db
from trading_bot.core.repository import MarketDataRepository
from trading_bot.data_ingestion import MarketDataProviderRegistry
from trading_bot.monitoring.logger import setup_logging

# Setup logger for the script
setup_logging()
logger = logging.getLogger("trading_bot.ingest_historical")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest historical market bars into SQL database."
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["yfinance", "ccxt"],
        required=True,
        help="The data source provider to use (yfinance or ccxt).",
    )
    parser.add_argument(
        "--ticker",
        "-t",
        required=True,
        help="The ticker or symbol to ingest (e.g. 'AAPL' for yfinance, 'BTC/USDT' for ccxt).",
    )
    parser.add_argument(
        "--exchange",
        "-e",
        default="binance",
        help="Exchange ID if using CCXT provider (default: 'binance').",
    )
    parser.add_argument(
        "--timeframe",
        "-f",
        default="1m",
        help="Data interval/timeframe (e.g., '1m', '5m', '1d').",
    )
    parser.add_argument(
        "--period",
        "-d",
        default="5d",
        help="Time period for yfinance download (e.g., '5d', '1mo', '1y').",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=500,
        help="Limit of bars to download (mostly for CCXT, default: 500).",
    )

    args = parser.parse_args()

    # 1. Initialize Database
    try:
        init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}")
        sys.exit(1)

    db = SessionLocal()
    repo = MarketDataRepository(db)

    try:
        # 2. Instantiate correct provider
        logger.info(f"Initializing provider '{args.provider}'...")
        try:
            provider_class = MarketDataProviderRegistry.get_provider_class(
                args.provider
            )

            provider = provider_class.from_args(args)
        except ImportError:
            sys.exit(1)
        except ValueError as e:
            logger.critical(str(e))
            sys.exit(1)

        # 3. Fetch details and ensure market exists in database
        logger.info(f"Fetching market details for ticker '{args.ticker}'...")
        details = provider.get_market_details(args.ticker)
        repo.ensure_market(details)
        logger.info(f"Market '{details.market_id}' registered in DB.")

        # 4. Fetch candles/bars
        logger.info(f"Downloading historical bars for '{args.ticker}'...")
        bars = provider.get_bars(args.ticker, count=args.limit)

        if not bars:
            logger.warning(f"No bars returned by provider for symbol '{args.ticker}'")
            return

        logger.info(
            f"Fetched {len(bars)} bars from provider. Inserting/upserting to SQL database..."
        )

        # 5. Store in SQL database
        inserted_count = repo.save_bars(args.ticker, bars)
        logger.info(
            f"Successfully stored data. New records added: {inserted_count} / Total fetched: {len(bars)}"
        )

    except Exception as e:
        logger.exception(f"Ingestion process failed: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
