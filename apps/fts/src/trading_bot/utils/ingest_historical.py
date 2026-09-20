# src/trading_bot/utils/ingest_historical.py

import argparse
import logging
import sys
from datetime import datetime, timezone

from trading_bot.core.database import SessionLocal, init_db
from trading_bot.core.dataset import calculate_dataset_hash
from trading_bot.core.repository import MarketDataRepository, ModelRepository
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
    parser.add_argument(
        "--until",
        "-u",
        type=str,
        default=None,
        help="Cutoff datetime in ISO format (e.g., '2026-06-01T03:30:00').",
    )
    parser.add_argument(
        "--since",
        "-s",
        type=str,
        default=None,
        help="Start datetime in ISO format (e.g., '2025-11-04T19:30:00').",
    )
    parser.add_argument(
        "--register-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Register combined bars as a TimeSeriesDataset in database (default: True).",
    )

    args = parser.parse_args()

    until_dt = None
    if isinstance(args.until, str):
        try:
            until_dt = datetime.fromisoformat(args.until)
            if until_dt.tzinfo is None:
                until_dt = until_dt.replace(tzinfo=timezone.utc)
        except ValueError as e:
            logger.critical(f"Invalid ISO datetime for --until parameter: {e}")
            sys.exit(1)
    elif isinstance(args.until, datetime):
        until_dt = args.until
        if until_dt.tzinfo is None:
            until_dt = until_dt.replace(tzinfo=timezone.utc)

    since_dt = None
    if isinstance(args.since, str):
        try:
            since_dt = datetime.fromisoformat(args.since)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except ValueError as e:
            logger.critical(f"Invalid ISO datetime for --since parameter: {e}")
            sys.exit(1)
    elif isinstance(args.since, datetime):
        since_dt = args.since
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)

    # 1. Initialize Database
    try:
        init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}")
        sys.exit(1)

    db = SessionLocal()
    repo = MarketDataRepository(db)
    model_repo = ModelRepository(db)

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
        fetch_kwargs = {"count": args.limit}
        if until_dt is not None:
            fetch_kwargs["until"] = until_dt
        if since_dt is not None:
            fetch_kwargs["since"] = since_dt

        bars = provider.get_bars(args.ticker, **fetch_kwargs)

        if not bars:
            logger.warning(f"No bars returned by provider for symbol '{args.ticker}'")
            return

        logger.info(
            f"Fetched {len(bars)} bars from provider. Inserting/upserting to SQL database..."
        )

        # 5. Store in SQL database
        inserted_count = repo.save_bars(args.ticker, bars)
        db.commit()
        logger.info(
            f"Successfully stored data. New records added: {inserted_count} / Total fetched: {len(bars)}"
        )

        # 6. Optional: Register concatenated TimeSeriesDataset in DB
        if args.register_dataset:
            all_bars_db = repo.get_bars(args.ticker, interval=args.timeframe)
            if all_bars_db:
                # Check for large gaps (> 4 * interval)
                gaps = []
                for i in range(1, len(all_bars_db)):
                    prev_ts = all_bars_db[i - 1].timestamp
                    curr_ts = all_bars_db[i].timestamp
                    gap_hours = (curr_ts - prev_ts).total_seconds() / 3600.0
                    if gap_hours > 6.0:  # Gap greater than 6 hours
                        gaps.append((prev_ts, curr_ts, gap_hours))

                if gaps:
                    logger.warning(
                        f"WARNING: Detected {len(gaps)} significant gap(s) in stored bars for {args.ticker}!"
                    )
                    for g_start, g_end, g_hrs in gaps:
                        logger.warning(
                            f"  Gap: {g_start} to {g_end} ({g_hrs:.1f} hours missing)"
                        )

                hash_val = calculate_dataset_hash(all_bars_db)
                dataset = model_repo.get_or_create_dataset(
                    market_id=args.ticker,
                    interval=args.timeframe,
                    start_time=all_bars_db[0].timestamp,
                    end_time=all_bars_db[-1].timestamp,
                    hash_val=hash_val,
                )
                db.commit()
                logger.info(
                    f"Registered TimeSeriesDataset ID: {dataset.dataset_id} | "
                    f"Total Bars: {len(all_bars_db)} | Period: {dataset.start_time} to {dataset.end_time}"
                )
                print(f"\n--- DATASET REGISTRATION SUCCESSFUL ---")
                print(f"Dataset ID: {dataset.dataset_id}")
                print(f"Market: {dataset.market_id} ({dataset.interval})")
                print(f"Total Bars: {len(all_bars_db)}")
                print(f"Start Time: {dataset.start_time}")
                print(f"End Time:   {dataset.end_time}")
                if gaps:
                    print(f"WARNING: {len(gaps)} gap(s) detected in database timeline!")
                print(f"Hash:       {dataset.hash}\n")

    except Exception as e:
        db.rollback()
        logger.exception(f"Ingestion process failed: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
