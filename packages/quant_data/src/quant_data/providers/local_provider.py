from typing import Any, Sequence

from quant_core.interfaces import BaseMarketDataProvider
from quant_core.models import BarData, Instrument, MarketDataRequest, OrderBook, Trade

from ..reference.gaps import GAP_RULES


class LocalCSVMarketDataProvider(BaseMarketDataProvider):
    gap_rule = GAP_RULES["LocalCSVMarketDataProvider"]

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

    def get_bars(self, request: MarketDataRequest) -> Sequence[BarData]:
        return []
