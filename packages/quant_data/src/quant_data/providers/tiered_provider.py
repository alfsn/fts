import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

from quant_core.interfaces import BaseMarketDataProvider
from quant_core.models import BarData, Instrument, MarketDataRequest, OrderBook, Trade
from quant_data.db.repositories import MarketDataRepository
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Longest gap between `start` and the first trading day (long weekends, holidays).
START_COVERAGE_TOLERANCE = timedelta(days=5)


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


class TieredDataProvider(BaseMarketDataProvider):
    """
    Implements a resilient, multi-provider waterfall using the Chain of Responsibility pattern.
    Acts as a drop-in replacement for any standard BaseMarketDataProvider.
    """

    @classmethod
    def from_args(cls, args: Any) -> "TieredDataProvider":
        raise NotImplementedError(
            "TieredDataProvider must be initialized with a db_session and provider_chain."
        )

    @classmethod
    def create_default(
        cls, db_session: Session, failing_tickers: Optional[set] = None
    ) -> "TieredDataProvider":
        """
        Builds the default tiered data provider waterfall.
        Tier 1: MockIBKR
        Tier 2: MockTiingo
        Tier 3: YFinance
        """
        from .mock_providers import MockIBKRProvider, MockTiingoProvider
        from .yfinance_provider import YFinanceMarketDataProvider

        failing_tickers = failing_tickers or set()
        chain = [
            MockIBKRProvider(failing_tickers=failing_tickers),
            MockTiingoProvider(failing_tickers=failing_tickers),
            YFinanceMarketDataProvider(),
        ]
        return cls(db_session, chain)

    def __init__(
        self, db_session: Session, provider_chain: Sequence[BaseMarketDataProvider]
    ):
        self.repo = MarketDataRepository(db_session)
        self.provider_chain = provider_chain

    def _cached_bars(
        self, request: MarketDataRequest, start: datetime, end: datetime
    ) -> list[BarData]:
        rows = self.repo.get_bars(
            request.instrument_id,
            start_date=start,
            end_date=end,
            interval=request.interval,
        )
        bars = [BarData.model_validate(r, from_attributes=True) for r in rows]
        for bar in bars:  # SQLite drops tzinfo; timestamps are stored as UTC.
            bar.timestamp = _as_utc(bar.timestamp)
        return bars

    @staticmethod
    def _covers_start(cached: Sequence[BarData], start: datetime) -> bool:
        """True if the cache reaches back to `start` (within a weekend/holiday gap)."""
        return cached[0].timestamp <= _as_utc(start) + START_COVERAGE_TOLERANCE

    def list_tradable_markets(self) -> Sequence[Instrument]:
        for provider in self.provider_chain:
            try:
                markets = provider.list_tradable_markets()
                if markets:
                    return markets
            except Exception as e:
                logger.warning(
                    f"{provider.__class__.__name__} failed list_tradable_markets: {e}"
                )
        return []

    def get_market_details(self, instrument_id: str) -> Instrument:
        for provider in self.provider_chain:
            try:
                details = provider.get_market_details(instrument_id)
                if details:
                    return details
            except Exception as e:
                logger.warning(
                    f"{provider.__class__.__name__} failed get_market_details for {instrument_id}: {e}"
                )
        raise ValueError(f"All providers failed to get details for {instrument_id}")

    def get_order_book(self, instrument_id: str) -> OrderBook:
        for provider in self.provider_chain:
            try:
                ob = provider.get_order_book(instrument_id)
                if ob and (ob.bids or ob.asks):
                    return ob
            except Exception as e:
                logger.warning(
                    f"{provider.__class__.__name__} failed get_order_book for {instrument_id}: {e}"
                )
        raise ValueError(f"All providers failed to get order book for {instrument_id}")

    def get_trade_history(self, instrument_id: str) -> Sequence[Trade]:
        for provider in self.provider_chain:
            try:
                trades = provider.get_trade_history(instrument_id)
                if trades:
                    return trades
            except Exception as e:
                logger.warning(
                    f"{provider.__class__.__name__} failed get_trade_history for {instrument_id}: {e}"
                )
        raise ValueError(
            f"All providers failed to get trade history for {instrument_id}"
        )

    def get_bars(self, request: MarketDataRequest) -> Sequence[BarData]:
        """
        Range requests (`start_date` set) are served from the DB cache and topped
        up incrementally: only bars after the last cached one are fetched.
        Count requests (no `start_date`) keep the "last `count` bars" semantics.
        """
        start, end = request.resolve_bounds()
        cached = self._cached_bars(request, start, end)

        if request.start_date is None:
            if len(cached) >= request.count:
                logger.info(f"Serving {request.instrument_id} bars from DB Cache.")
                return cached[-request.count :]
            data = self._fetch_and_cache(request)
            if not data:
                raise self._all_failed(request)
            return data

        if cached and self._covers_start(cached, start):
            # Top up: only bars from the last cached one onwards.
            request_tail = request.model_copy(
                update={"start_date": cached[-1].timestamp}
            )
            if self._fetch_and_cache(request_tail):
                cached = self._cached_bars(request, start, end)
            else:
                logger.info(f"Serving {request.instrument_id} bars from DB Cache.")
            return cached

        if not self._fetch_and_cache(request):
            if not cached:
                raise self._all_failed(request)
            return cached
        return self._cached_bars(request, start, end)

    def _fetch_and_cache(self, request: MarketDataRequest) -> Sequence[BarData]:
        data = self._fetch_from_chain(request)
        if data:
            self.repo.save_bars(request.instrument_id, data)
            self.repo.db.commit()
        return data

    @staticmethod
    def _all_failed(request: MarketDataRequest) -> ValueError:
        return ValueError(
            f"All providers in the waterfall failed to get bars for {request.instrument_id}"
        )

    def _fetch_from_chain(self, request: MarketDataRequest) -> Sequence[BarData]:
        for provider in self.provider_chain:
            try:
                data = provider.get_bars(request)
                if data:
                    logger.info(
                        f"Serving {request.instrument_id} bars from {provider.__class__.__name__}."
                    )
                    return data
            except Exception as e:
                logger.warning(
                    f"Provider {provider.__class__.__name__} failed get_bars for {request.instrument_id}: {e}"
                )
        return []
