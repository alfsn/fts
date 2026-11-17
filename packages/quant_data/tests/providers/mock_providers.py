import logging
from datetime import datetime, timezone
from typing import Any, Sequence, Set

from quant_core.enums import AssetType, BarType, Geography
from quant_core.interfaces import BaseMarketDataProvider
from quant_core.models import (
    BarData,
    Instrument,
    MarketDataRequest,
    OrderBook,
    Trade,
    TradFiDetails,
)

logger = logging.getLogger(__name__)


class MockFailingProvider(BaseMarketDataProvider):
    """Base mock provider that fails for specific tickers."""

    @classmethod
    def from_args(cls, args: Any) -> "MockFailingProvider":
        return cls()

    def __init__(self, name: str, failing_tickers: Set[str] = None):
        self.name = name
        self.failing_tickers = failing_tickers or set()

    def _check_fail(self, instrument_id: str):
        if instrument_id in self.failing_tickers:
            raise ValueError(f"{self.name} simulated failure for {instrument_id}")

    def list_tradable_markets(self) -> Sequence[Instrument]:
        return []

    def get_market_details(self, instrument_id: str) -> Instrument:
        self._check_fail(instrument_id)
        return Instrument(
            instrument_id=instrument_id,
            name=f"{self.name} Mock {instrument_id}",
            details=TradFiDetails(
                asset_type=AssetType.STOCK,
                geography=Geography.US,
                industry="Mock",
                currency="USD",
            ),
        )

    def get_order_book(self, instrument_id: str) -> OrderBook:
        self._check_fail(instrument_id)
        return OrderBook(bids=[], asks=[], timestamp=datetime.now(timezone.utc))

    def get_trade_history(self, instrument_id: str) -> Sequence[Trade]:
        self._check_fail(instrument_id)
        return []

    def get_bars(self, request: MarketDataRequest) -> Sequence[BarData]:
        self._check_fail(request.instrument_id)
        # Returns dummy data on success so it doesn't trigger a fallback
        return [
            BarData(
                timestamp=datetime.now(timezone.utc),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0,
                bar_type=BarType.TIME,
                interval=request.interval,
                ticks_count=1,
                dollar_volume=102000.0,
            )
        ]


class MockIBKRProvider(MockFailingProvider):
    def __init__(self, failing_tickers: Set[str] = None):
        super().__init__("MockIBKR", failing_tickers)


class MockTiingoProvider(MockFailingProvider):
    def __init__(self, failing_tickers: Set[str] = None):
        super().__init__("MockTiingo", failing_tickers)
