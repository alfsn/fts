# tests/unit/test_ccxt_plugin.py

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from ccxt_plugin.data_providers import CCXTMarketDataProvider

from trading_bot.core.enums import BarType, OrderSide


def test_ccxt_market_details():
    provider = CCXTMarketDataProvider(exchange_id="binance")
    details = provider.get_market_details("BTC/USDT")

    assert details.market_id == "BTC/USDT"
    assert "BINANCE" in details.name
    assert details.resolution_source == "binance"


@patch("ccxt_plugin.data_providers.ccxt.binance")
def test_ccxt_get_bars(mock_binance):
    # Setup mock exchange instance
    mock_exchange = MagicMock()
    mock_binance.return_value = mock_exchange

    # Mock fetch_ohlcv returning: [[timestamp, open, high, low, close, volume]]
    mock_exchange.fetch_ohlcv.return_value = [
        [1781568000000, 50000.0, 50100.0, 49900.0, 50050.0, 10.0],
        [1781568060000, 50050.0, 50200.0, 50000.0, 50150.0, 15.0],
    ]

    provider = CCXTMarketDataProvider(exchange_id="binance", timeframe="1m")
    bars = provider.get_bars("BTC/USDT")

    assert len(bars) == 2
    assert bars[0].close == 50050.0
    assert bars[1].close == 50150.0
    assert bars[0].volume == 10.0
    assert bars[0].bar_type == BarType.TIME
    assert bars[0].timestamp == datetime.fromtimestamp(1781568000.0, tz=timezone.utc)


@patch("ccxt_plugin.data_providers.ccxt.binance")
def test_ccxt_get_order_book(mock_binance):
    mock_exchange = MagicMock()
    mock_binance.return_value = mock_exchange

    mock_exchange.fetch_order_book.return_value = {
        "timestamp": 1781568060000,
        "bids": [[50000.0, 1.5], [49950.0, 2.0]],
        "asks": [[50050.0, 0.8], [50100.0, 1.2]],
    }

    provider = CCXTMarketDataProvider(exchange_id="binance")
    ob = provider.get_order_book("BTC/USDT")

    assert len(ob.bids) == 2
    assert len(ob.asks) == 2
    assert ob.bids[0].price == 50000.0
    assert ob.bids[0].size == 1.5
    assert ob.asks[0].price == 50050.0


@patch("ccxt_plugin.data_providers.ccxt.binance")
def test_ccxt_get_trade_history(mock_binance):
    mock_exchange = MagicMock()
    mock_binance.return_value = mock_exchange

    mock_exchange.fetch_trades.return_value = [
        {
            "id": "12345",
            "timestamp": 1781568060000,
            "price": 50025.0,
            "amount": 0.5,
            "side": "buy",
        },
        {
            "id": "12346",
            "timestamp": 1781568065000,
            "price": 50020.0,
            "amount": 0.3,
            "side": "sell",
        },
    ]

    provider = CCXTMarketDataProvider(exchange_id="binance")
    trades = provider.get_trade_history("BTC/USDT")

    assert len(trades) == 2
    assert trades[0].price == 50025.0
    assert trades[0].side == OrderSide.BUY
    assert trades[1].side == OrderSide.SELL


@patch("ccxt_plugin.data_providers.ccxt.binance")
def test_ccxt_get_bars_paginated(mock_binance):
    mock_exchange = MagicMock()
    mock_binance.return_value = mock_exchange
    mock_exchange.parse_timeframe.return_value = 1800  # 30m in seconds

    # Simulate two batches returned by fetch_ohlcv
    batch1 = [
        [1700000000000 + i * 1800000, 100.0, 105.0, 95.0, 102.0, 10.0]
        for i in range(1000)
    ]
    batch2 = [
        [1700000000000 + (1000 + i) * 1800000, 102.0, 106.0, 98.0, 104.0, 12.0]
        for i in range(500)
    ]
    mock_exchange.fetch_ohlcv.side_effect = [batch1, batch2]

    provider = CCXTMarketDataProvider(exchange_id="binance", timeframe="30m")
    until_dt = datetime.fromtimestamp(
        (1700000000000 + 1500 * 1800000) / 1000.0, tz=timezone.utc
    )
    bars = provider.get_bars("BTC/USDT", count=1500, until=until_dt)

    assert len(bars) == 1500
    assert mock_exchange.fetch_ohlcv.call_count == 2
