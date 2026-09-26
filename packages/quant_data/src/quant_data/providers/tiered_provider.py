import logging
from datetime import datetime
from typing import Any, Optional, Sequence

from quant_core.interfaces import BaseMarketDataProvider
from quant_core.models import BarData, Instrument, MarketDataRequest, OrderBook, Trade
from quant_data.db.repositories import MarketDataRepository
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


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

    def _is_data_complete(
        self,
        cached_data: Sequence[BarData],
        count: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> bool:
        """
        Determines if the cached data satisfies the request requirements.
        """
        if not cached_data:
            return False

        if len(cached_data) >= count:
            return True

        return False

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
        start, end = request.resolve_bounds()

        # 1. Check DB Cache
        cached_data = self.repo.get_bars(
            request.instrument_id,
            start_date=start,
            end_date=end,
            interval=request.interval,
        )
        if self._is_data_complete(cached_data, request.count, start, end):
            logger.info(f"Serving {request.instrument_id} bars from DB Cache.")
            return cached_data[-request.count :] if request.count else cached_data

        # 2. Tiered Fallback
        for provider in self.provider_chain:
            try:
                data = provider.get_bars(request)
                if data:
                    logger.info(
                        f"Serving {request.instrument_id} bars from {provider.__class__.__name__}."
                    )
                    # Cache in DB for next time
                    self.repo.save_bars(request.instrument_id, data)
                    self.repo.db.commit()
                    return data
            except Exception as e:
                logger.warning(
                    f"Provider {provider.__class__.__name__} failed get_bars for {request.instrument_id}: {e}"
                )
                continue

        raise ValueError(
            f"All providers in the waterfall failed to get bars for {request.instrument_id}"
        )
