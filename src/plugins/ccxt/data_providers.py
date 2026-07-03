# src/plugins/ccxt/data_providers.py

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

import ccxt

from trading_bot.core.enums import BarType, OrderSide
from trading_bot.core.schemas import (
    BarData,
    MarketData,
    MarketDetails,
    OrderBook,
    PriceLevel,
    Trade,
)
from trading_bot.data_ingestion.abc import BaseMarketDataProvider

logger = logging.getLogger(__name__)


class CCXTMarketDataProvider(BaseMarketDataProvider):
    """
    Concrete market data provider that fetches OHLCV candles, order books,
    and recent trade logs from crypto exchanges using the CCXT library.
    """

    @classmethod
    def from_args(cls, args: Any) -> "CCXTMarketDataProvider":
        """
        Creates an instance of the provider from parsed command-line arguments.
        """
        return cls(exchange_id=args.exchange, timeframe=args.timeframe)

    def __init__(
        self,
        exchange_id: str = "binance",
        timeframe: str = "1m",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the CCXT exchange integration.

        :param exchange_id: The identifier of the exchange in CCXT (e.g. 'binance', 'coinbase').
        :param timeframe: The candle interval timeframe (e.g. '1m', '5m', '1h', '1d').
        :param config: Dictionary of configuration settings to pass to CCXT (e.g., API keys, rateLimit).
        """
        self.exchange_id = exchange_id.lower()
        self.timeframe = timeframe
        self.config = config or {}

        # Instantiate CCXT exchange class dynamically
        if not hasattr(ccxt, self.exchange_id):
            raise ValueError(f"Exchange '{self.exchange_id}' is not supported by ccxt.")

        exchange_class = getattr(ccxt, self.exchange_id)
        # Enable rate limiting by default
        if "enableRateLimit" not in self.config:
            self.config["enableRateLimit"] = True
        self.exchange = exchange_class(self.config)

    def list_tradable_markets(self) -> Sequence[MarketDetails]:
        """
        Fetches all active markets from the exchange.
        """
        try:
            markets = self.exchange.load_markets()
            results = []
            for symbol, info in markets.items():
                # Filter active markets if the exchange provides that flag
                if info.get("active", True):
                    results.append(
                        MarketDetails(
                            market_id=symbol,
                            name=f"{self.exchange_id.upper()} {symbol}",
                            end_date=datetime.max.replace(tzinfo=timezone.utc),
                            resolution_source=self.exchange_id,
                        )
                    )
            return results
        except Exception as e:
            logger.error(f"Failed to load markets for exchange {self.exchange_id}: {e}")
            return []

    def get_market_details(self, market_id: str) -> MarketDetails:
        """
        Returns static information for a single symbol.
        """
        return MarketDetails(
            market_id=market_id,
            name=f"{self.exchange_id.upper()} {market_id}",
            end_date=datetime.max.replace(tzinfo=timezone.utc),
            resolution_source=self.exchange_id,
        )

    def get_order_book(self, market_id: str) -> OrderBook:
        """
        Fetches the current L2 order book from the exchange.
        """
        try:
            raw_ob = self.exchange.fetch_order_book(market_id)
            timestamp = datetime.fromtimestamp(
                (raw_ob.get("timestamp") or self.exchange.milliseconds()) / 1000.0,
                tz=timezone.utc,
            )

            bids = [
                PriceLevel(price=float(bid[0]), size=float(bid[1]))
                for bid in raw_ob.get("bids", [])
            ]
            asks = [
                PriceLevel(price=float(ask[0]), size=float(ask[1]))
                for ask in raw_ob.get("asks", [])
            ]

            return OrderBook(
                bids=bids,
                asks=asks,
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch order book for {market_id} from {self.exchange_id}: {e}"
            )
            return OrderBook(
                bids=[],
                asks=[],
            )

    def get_trade_history(self, market_id: str) -> Sequence[Trade]:
        """
        Fetches recent public trades for a symbol.
        """
        try:
            raw_trades = self.exchange.fetch_trades(market_id)
            trades = []
            for t in raw_trades:
                ts = datetime.fromtimestamp(t["timestamp"] / 1000.0, tz=timezone.utc)
                side_str = t.get("side")
                side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

                trades.append(
                    Trade(
                        timestamp=ts,
                        price=float(t["price"]),
                        size=float(t["amount"]),
                        side=side,
                    )
                )
            return trades
        except Exception as e:
            logger.error(
                f"Failed to fetch trade history for {market_id} from {self.exchange_id}: {e}"
            )
            return []

    def get_bars(
        self,
        market_id: str,
        count: int = 100,
        until: Optional[datetime] = None,
        since: Optional[datetime] = None,
    ) -> Sequence[BarData]:
        """
        Downloads historical candlesticks (OHLCV) from the exchange in batches.

        :param market_id: Symbol identifier (e.g. 'BTC/USDT').
        :param count: Total number of bars requested.
        :param until: Optional cutoff datetime (inclusive end).
        :param since: Optional starting datetime (inclusive start).
        """
        try:
            # Determine timeframe duration in milliseconds
            if hasattr(self.exchange, "parse_timeframe"):
                tf_sec = self.exchange.parse_timeframe(self.timeframe)
            else:
                tf_sec = 1800  # Default to 30m if unknown
            tf_ms = tf_sec * 1000

            until_utc = (
                until.astimezone(timezone.utc)
                if until and until.tzinfo
                else (until.replace(tzinfo=timezone.utc) if until else None)
            )
            since_utc = (
                since.astimezone(timezone.utc)
                if since and since.tzinfo
                else (since.replace(tzinfo=timezone.utc) if since else None)
            )

            until_ms = int(until_utc.timestamp() * 1000) if until_utc else None

            if since_utc is not None:
                since_ms = int(since_utc.timestamp() * 1000)
            elif until_ms is not None:
                since_ms = until_ms - (count * tf_ms)
            else:
                since_ms = None

            # Single-batch optimization for small non-time-bounded queries
            if count <= 1000 and since_ms is None and until_ms is None:
                ohlcv_data = [
                    self.exchange.fetch_ohlcv(
                        market_id, timeframe=self.timeframe, limit=count
                    )
                ]
            else:
                # Paginated multi-batch fetching
                ohlcv_data = []
                current_since = since_ms
                last_fetched_ts = -1

                while sum(len(b) for b in ohlcv_data) < count:
                    batch_limit = min(1000, count - sum(len(b) for b in ohlcv_data))
                    fetch_kwargs: Dict[str, Any] = {"limit": batch_limit}
                    if current_since is not None:
                        fetch_kwargs["since"] = current_since

                    logger.info(
                        f"Fetching CCXT batch: symbol={market_id}, since={current_since}, limit={batch_limit}"
                    )
                    batch = self.exchange.fetch_ohlcv(
                        market_id, timeframe=self.timeframe, **fetch_kwargs
                    )

                    if not batch:
                        logger.info("No more bars returned by CCXT exchange.")
                        break

                    # Check for progress to avoid infinite loop
                    latest_ts = batch[-1][0]
                    if latest_ts <= last_fetched_ts:
                        break

                    ohlcv_data.append(batch)
                    last_fetched_ts = latest_ts
                    current_since = latest_ts + tf_ms

                    if until_ms is not None and current_since > until_ms:
                        break

                    if len(batch) < batch_limit:
                        break

            bars = []
            seen_timestamps = set()

            for batch in ohlcv_data:
                for candle in batch:
                    dt = datetime.fromtimestamp(candle[0] / 1000.0, tz=timezone.utc)

                    if until_utc and dt > until_utc:
                        continue
                    if since_utc and dt < since_utc:
                        continue

                    if dt in seen_timestamps:
                        continue
                    seen_timestamps.add(dt)

                    op = float(candle[1])
                    hi = float(candle[2])
                    lo = float(candle[3])
                    cl = float(candle[4])
                    vol = float(candle[5])

                    if op <= 0 or hi <= 0 or lo <= 0 or cl <= 0:
                        continue

                    bars.append(
                        BarData(
                            timestamp=dt,
                            open=op,
                            high=hi,
                            low=lo,
                            close=cl,
                            volume=vol,
                            bar_type=BarType.TIME,
                            interval=self.timeframe,
                            ticks_count=1,
                            dollar_volume=cl * vol,
                        )
                    )

            if len(bars) > count:
                bars = bars[-count:]

            return bars
        except Exception as e:
            logger.error(
                f"Failed to fetch bars for {market_id} from {self.exchange_id}: {e}"
            )
            return []
