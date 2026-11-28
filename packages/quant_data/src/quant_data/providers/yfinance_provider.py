import logging
from datetime import datetime, timezone
from typing import Any, Dict, Sequence

import pandas as pd
import yfinance as yf
from quant_core.analytics import local_dates, unadjust_dividends, unadjust_splits
from quant_core.enums import AssetType, BarType, CorporateActionType, Geography
from quant_core.interfaces import (
    BaseMarketDataProvider,
    CorporateActionProvider,
    RateLimitedError,
)
from quant_core.models import (
    BarData,
    CorporateAction,
    Instrument,
    MarketDataRequest,
    OrderBook,
    RateLimit,
    Trade,
    TradFiDetails,
)
from yfinance.exceptions import YFRateLimitError

logger = logging.getLogger(__name__)


def _utc_midnight(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local calendar dates keyed at 00:00 UTC, the same key daily bars use."""
    return local_dates(index).tz_localize("UTC")


class YFinanceMarketDataProvider(BaseMarketDataProvider, CorporateActionProvider):
    """
    OHLCV bars and corporate actions from Yahoo Finance (yfinance).

    Yahoo rewrites history: ``Close``, ``Volume`` and dividend amounts are all
    split-adjusted retroactively, even with ``auto_adjust=False``. This adapter
    undoes that with the symbol's full split history, so ``get_bars`` returns
    prices as traded for every interval and ``get_corporate_actions`` returns
    dividends as paid (PR 2a, D2/D14).
    """

    # Yahoo publishes no quota and blocks bursts with HTTP 429 (yfinance #2422).
    rate_limit = RateLimit(min_interval_s=1.0, max_retries=3, backoff_s=30.0)

    @classmethod
    def from_args(cls, args: Any) -> "YFinanceMarketDataProvider":
        return cls(default_interval=args.timeframe)

    def __init__(self, default_interval: str = "1d") -> None:
        self.default_interval = default_interval
        self._tickers: Dict[str, Any] = {}
        self._splits: Dict[str, list[CorporateAction]] = {}

    def list_tradable_markets(self) -> Sequence[Instrument]:
        """Yahoo has no symbol directory; a few well-known tickers."""
        default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
        return [self.get_market_details(ticker) for ticker in default_tickers]

    def get_market_details(self, instrument_id: str) -> Instrument:
        return Instrument(
            instrument_id=instrument_id,
            name=f"{instrument_id} Stock",
            details=TradFiDetails(
                asset_type=AssetType.STOCK,
                geography=Geography.US,
                industry="Tech",
                currency="USD",
            ),
        )

    def get_order_book(self, instrument_id: str) -> OrderBook:
        logger.warning(
            f"get_order_book not supported by Yahoo Finance for {instrument_id}"
        )
        return OrderBook(bids=[], asks=[], timestamp=datetime.now(timezone.utc))

    def get_trade_history(self, instrument_id: str) -> Sequence[Trade]:
        logger.warning(
            f"get_trade_history not supported by Yahoo Finance for {instrument_id}"
        )
        return []

    # --- corporate actions -------------------------------------------------

    def _ticker(self, symbol: str):
        """One ``yf.Ticker`` per symbol: ``.splits`` and ``.dividends`` share a
        single ``history(period="max")`` request."""
        if symbol not in self._tickers:
            self._tickers[symbol] = yf.Ticker(symbol)
        return self._tickers[symbol]

    def _split_actions(self, symbol: str) -> list[CorporateAction]:
        if symbol not in self._splits:
            splits = _rate_limited(lambda: self._ticker(symbol).splits)
            self._splits[symbol] = [
                CorporateAction(
                    instrument_id=symbol,
                    action_type=CorporateActionType.SPLIT,
                    ex_date=ex.to_pydatetime(),
                    ratio=float(ratio),
                )
                for ex, ratio in zip(_utc_midnight(splits.index), splits.to_numpy())
            ]
        return self._splits[symbol]

    def get_corporate_actions(self, instrument_id: str) -> Sequence[CorporateAction]:
        splits = self._split_actions(instrument_id)
        dividends = _rate_limited(lambda: self._ticker(instrument_id).dividends)
        raw = unadjust_dividends(dividends, splits)
        return splits + [
            CorporateAction(
                instrument_id=instrument_id,
                action_type=CorporateActionType.DIVIDEND,
                ex_date=ex.to_pydatetime(),
                amount=float(amount),
            )
            for ex, amount in zip(_utc_midnight(raw.index), raw.to_numpy())
        ]

    # --- bars ----------------------------------------------------------------

    def get_bars(self, request: MarketDataRequest) -> Sequence[BarData]:
        interval = request.interval or self.default_interval
        start, end = request.resolve_bounds()

        df = _rate_limited(
            lambda: yf.download(
                request.instrument_id,
                start=start,
                end=end,
                interval=interval,
                # Yahoo's unadjusted frame: no dividend adjustment, but splits
                # are still applied retroactively, hence unadjust_splits below.
                auto_adjust=False,
                progress=False,
            )
        )
        if df.empty:
            logger.warning(
                f"No data returned from yfinance for market {request.instrument_id}"
            )
            return []

        # Flatten MultiIndex columns if present (common in newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Undo the split adjustment while the index is still in the exchange's
        # local frame (tz-naive dates for 1d, tz-aware local for intraday).
        df = unadjust_splits(df, self._split_actions(request.instrument_id))

        # A count applies only when no explicit range was requested.
        df_subset = df if request.start_date else df.tail(request.count)
        bars = []
        skipped = 0

        for ts, row in df_subset.iterrows():
            op = float(row["Open"])
            hi = float(row["High"])
            lo = float(row["Low"])
            cl = float(row["Close"])
            vol = float(row["Volume"])

            # Enforce schema constraints (gt=0 for prices, ge=0 for volume).
            # Written so NaN fails too: Yahoo sends NaN rows (or NaN volume)
            # for some BYMA sessions, and one would reject the whole download.
            # A missing value is dropped, never filled; the count is logged.
            if not (op > 0 and hi > 0 and lo > 0 and cl > 0 and vol >= 0):
                skipped += 1
                continue

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
                    interval=interval,
                    ticks_count=1,
                    dollar_volume=cl * vol,
                )
            )

        if skipped:
            logger.warning(
                "%s: skipped %d Yahoo rows with a missing price or volume.",
                request.instrument_id,
                skipped,
            )
        return bars


def _rate_limited(fn):
    """Translate yfinance's 429 into the shared error the throttle retries."""
    try:
        return fn()
    except YFRateLimitError as e:
        raise RateLimitedError(str(e)) from e
