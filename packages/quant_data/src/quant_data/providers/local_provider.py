from typing import Any, Sequence

from quant_core.interfaces import BaseMarketDataProvider
from quant_core.models import BarData, Instrument, OrderBook, Trade


class LocalCSVMarketDataProvider(BaseMarketDataProvider):
    @classmethod
    def from_args(cls, args: Any) -> "BaseMarketDataProvider":
        return cls()

    def list_tradable_markets(self) -> Sequence[Instrument]:
        return []

    def get_market_details(self, instrument_id: str) -> Instrument:
        raise NotImplementedError()

    def get_order_book(self, instrument_id: str) -> OrderBook:
        raise NotImplementedError()

    def get_trade_history(self, instrument_id: str) -> Sequence[Trade]:
        return []

    def get_bars(self, instrument_id: str, count: int = 100) -> Sequence[BarData]:
        return []
