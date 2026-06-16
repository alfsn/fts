# tests/unit/test_ingest_historical.py

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from trading_bot.core.enums import BarType
from trading_bot.core.schemas import BarData, MarketDetails
from trading_bot.utils.ingest_historical import main


@patch("trading_bot.utils.ingest_historical.argparse.ArgumentParser.parse_args")
@patch("trading_bot.utils.ingest_historical.init_db")
@patch("trading_bot.utils.ingest_historical.SessionLocal")
@patch("trading_bot.utils.ingest_historical.MarketDataRepository")
def test_ingest_historical_yfinance(
    mock_repo_cls, mock_session_local, mock_init_db, mock_parse_args
):
    # 1. Setup argparse mock
    mock_args = MagicMock()
    mock_args.provider = "yfinance"
    mock_args.ticker = "AAPL"
    mock_args.timeframe = "1m"
    mock_args.period = "5d"
    mock_args.limit = 10
    mock_parse_args.return_value = mock_args

    # 2. Setup mock DB sessions and repos
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db
    mock_repo = MagicMock()
    mock_repo_cls.return_value = mock_repo

    # 3. Mock yfinance provider client
    mock_provider = MagicMock()
    mock_provider.get_market_details.return_value = MarketDetails(
        market_id="AAPL",
        name="Apple Inc",
        end_date=datetime.now(timezone.utc),
        resolution_source="yfinance",
    )
    mock_bars = [
        BarData(
            timestamp=datetime.now(timezone.utc),
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            volume=100.0,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=15050.0,
        )
    ]
    mock_provider.get_bars.return_value = mock_bars

    mock_provider_class = MagicMock()
    mock_provider_class.from_args.return_value = mock_provider
    with patch(
        "trading_bot.utils.ingest_historical.MarketDataProviderRegistry.get_provider_class",
        return_value=mock_provider_class,
    ):
        main()

    # 4. Verify mocks were called correctly
    mock_init_db.assert_called_once()
    mock_repo_cls.assert_called_once_with(mock_db)
    mock_provider.get_market_details.assert_called_once_with("AAPL")
    mock_repo.ensure_market.assert_called_once()
    mock_provider.get_bars.assert_called_once_with("AAPL", count=10)
    mock_repo.save_bars.assert_called_once_with("AAPL", mock_bars)
    mock_db.close.assert_called_once()


@patch("trading_bot.utils.ingest_historical.argparse.ArgumentParser.parse_args")
@patch("trading_bot.utils.ingest_historical.init_db")
@patch("trading_bot.utils.ingest_historical.SessionLocal")
@patch("trading_bot.utils.ingest_historical.MarketDataRepository")
def test_ingest_historical_ccxt(
    mock_repo_cls, mock_session_local, mock_init_db, mock_parse_args
):
    # 1. Setup argparse mock
    mock_args = MagicMock()
    mock_args.provider = "ccxt"
    mock_args.ticker = "BTC/USDT"
    mock_args.exchange = "binance"
    mock_args.timeframe = "1m"
    mock_args.limit = 50
    mock_parse_args.return_value = mock_args

    # 2. Setup mock DB sessions and repos
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db
    mock_repo = MagicMock()
    mock_repo_cls.return_value = mock_repo

    # 3. Mock CCXT provider client
    mock_provider = MagicMock()
    mock_provider.get_market_details.return_value = MarketDetails(
        market_id="BTC/USDT",
        name="Binance BTC/USDT",
        end_date=datetime.max.replace(tzinfo=timezone.utc),
        resolution_source="binance",
    )
    mock_bars = [
        BarData(
            timestamp=datetime.now(timezone.utc),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=5.0,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=250250.0,
        )
    ]
    mock_provider.get_bars.return_value = mock_bars

    mock_provider_class = MagicMock()
    mock_provider_class.from_args.return_value = mock_provider
    with patch(
        "trading_bot.utils.ingest_historical.MarketDataProviderRegistry.get_provider_class",
        return_value=mock_provider_class,
    ):
        main()

    # 4. Verify mocks were called correctly
    mock_init_db.assert_called_once()
    mock_repo_cls.assert_called_once_with(mock_db)
    mock_provider.get_market_details.assert_called_once_with("BTC/USDT")
    mock_repo.ensure_market.assert_called_once()
    mock_provider.get_bars.assert_called_once_with("BTC/USDT", count=50)
    mock_repo.save_bars.assert_called_once_with("BTC/USDT", mock_bars)
    mock_db.close.assert_called_once()
