# tests/unit/test_yfinance_plugin.py

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest
from yfinance_plugin.data_providers import YFinanceMarketDataProvider

from trading_bot.core.enums import BarType


def test_yfinance_market_details():
    provider = YFinanceMarketDataProvider()
    details = provider.get_market_details("AAPL")

    assert details.market_id == "AAPL"
    assert "AAPL" in details.name
    assert details.resolution_source == "yfinance"


@patch("yfinance_plugin.data_providers.yf.download")
def test_yfinance_get_bars(mock_download):
    # Setup mock dataframe from yfinance
    dates = pd.date_range(start="2026-06-10 09:30:00", periods=3, freq="min", tz="UTC")
    data = {
        "Open": [150.0, 151.0, 152.0],
        "High": [150.5, 151.5, 152.5],
        "Low": [149.5, 150.5, 151.5],
        "Close": [150.2, 151.2, 152.2],
        "Volume": [1000.0, 1100.0, 1200.0],
    }
    mock_df = pd.DataFrame(data, index=dates)
    mock_download.return_value = mock_df

    provider = YFinanceMarketDataProvider(period="5d", interval="1m")
    bars = provider.get_bars("AAPL", count=2)

    # We requested count=2, so it should return the last 2 items
    assert len(bars) == 2
    assert bars[0].close == 151.2
    assert bars[1].close == 152.2
    assert bars[0].volume == 1100.0
    assert bars[1].bar_type == BarType.TIME
    assert bars[0].timestamp == datetime(2026, 6, 10, 9, 31, tzinfo=timezone.utc)
    assert bars[1].timestamp == datetime(2026, 6, 10, 9, 32, tzinfo=timezone.utc)


@patch("yfinance_plugin.data_providers.yf.download")
def test_yfinance_get_bars_handles_zeros(mock_download):
    # Setup mock dataframe containing a zero price row (should be filtered out)
    dates = pd.date_range(start="2026-06-10 09:30:00", periods=2, freq="min", tz="UTC")
    data = {
        "Open": [0.0, 151.0],
        "High": [150.5, 151.5],
        "Low": [149.5, 150.5],
        "Close": [150.2, 151.2],
        "Volume": [1000.0, 1100.0],
    }
    mock_df = pd.DataFrame(data, index=dates)
    mock_download.return_value = mock_df

    provider = YFinanceMarketDataProvider()
    bars = provider.get_bars("AAPL")

    # The first row with Open=0.0 should be filtered out
    assert len(bars) == 1
    assert bars[0].close == 151.2
