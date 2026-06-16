# src/plugins/yfinance/data_providers.py

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd
import yfinance as yf

from trading_bot.core.enums import BarType
from trading_bot.core.schemas import (
    BarData,
    MarketData,
    MarketDetails,
    OrderBook,
    Trade,
)
from trading_bot.data_ingestion.abc import BaseMarketDataProvider

logger = logging.getLogger(__name__)


class YFinanceMarketDataProvider(BaseMarketDataProvider):
    """
    Concrete market data provider that fetches OHLCV candles
    from Yahoo Finance (yfinance) for stocks, ETFs, indices, etc.
    """

    @classmethod
    def from_args(cls, args: Any) -> "YFinanceMarketDataProvider":
        """
        Creates an instance of the provider from parsed command-line arguments.
        """
        return cls(period=args.period, interval=args.timeframe)

    def __init__(self, period: str = "5d", interval: str = "1m") -> None:
        """
        Initializes the Yahoo Finance data provider.

        :param period: The historical time period to download (e.g. '5d', '1mo').
        :param interval: The frequency interval (e.g. '1m', '5m', '1h', '1d').
        """
        self.period = period
        self.interval = interval

    def list_tradable_markets(self) -> Sequence[MarketDetails]:
        """
        Lists some default example tickers.
        Yahoo Finance doesn't have a list of all tickers.
        """
        default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
        return [self.get_market_details(ticker) for ticker in default_tickers]

    def get_market_details(self, market_id: str) -> MarketDetails:
        """
        Returns static information for the stock ticker.
        """
        return MarketDetails(
            market_id=market_id,
            name=f"{market_id} Stock",
            end_date=datetime.max.replace(tzinfo=timezone.utc),
            resolution_source="yfinance",
        )

    def get_order_book(self, market_id: str) -> OrderBook:
        """
        Yahoo Finance REST API does not support L2 order books.
        Returns a dummy empty order book.
        """
        logger.warning(f"get_order_book not supported by Yahoo Finance for {market_id}")
        return OrderBook(
            bids=[],
            asks=[],
            timestamp=datetime.now(timezone.utc),
        )

    def get_trade_history(self, market_id: str) -> Sequence[Trade]:
        """
        Yahoo Finance REST API does not support real-time tick trade logs.
        """
        logger.warning(
            f"get_trade_history not supported by Yahoo Finance for {market_id}"
        )
        return []

    def get_bars(self, market_id: str, count: int = 100) -> Sequence[BarData]:
        """
        Downloads the latest bars from yfinance and parses them into BarData.
        """
        # Download data using yfinance
        df = yf.download(
            market_id, period=self.period, interval=self.interval, progress=False
        )
        if df.empty:
            logger.warning(f"No data returned from yfinance for market {market_id}")
            return []

        # Flatten MultiIndex columns if present (common in newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure we are looking at the last 'count' rows
        df_subset = df.tail(count)
        bars = []

        for ts, row in df_subset.iterrows():
            # Parse prices
            op = float(row["Open"])
            hi = float(row["High"])
            lo = float(row["Low"])
            cl = float(row["Close"])
            vol = float(row["Volume"])

            # Enforce schemas constraints (gt=0 for prices)
            if op <= 0 or hi <= 0 or lo <= 0 or cl <= 0:
                continue

            # Convert timestamp to timezone-aware datetime
            # yfinance index can be datetime index or date index
            if hasattr(ts, "to_pydatetime"):
                dt = ts.to_pydatetime()
            else:
                dt = datetime.combine(ts, datetime.min.time())

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)

            bars.append(
                BarData(
                    timestamp=dt,
                    open=op,
                    high=hi,
                    low=lo,
                    close=cl,
                    volume=vol,
                    bar_type=BarType.TIME,
                    ticks_count=1,
                    dollar_volume=cl * vol,
                )
            )

        return bars

    def get_market_data(self, market_id: str) -> MarketData:
        """
        Returns a composite snapshot of the latest data for a market.
        """
        details = self.get_market_details(market_id)
        bars = list(self.get_bars(market_id))
        return MarketData(
            market_id=market_id,
            details=details,
            recent_bars=bars,
            order_book=None,
            recent_trades=None,
        )
