from abc import ABC, abstractmethod
from typing import Any, Sequence

from .models import BarData, Instrument, MarketData, OrderBook, Trade


class BaseMarketDataProvider(ABC):
    @classmethod
    def from_args(cls, args: Any) -> "BaseMarketDataProvider":
        raise NotImplementedError()

    @abstractmethod
    def list_tradable_markets(self) -> Sequence[Instrument]:
        pass

    @abstractmethod
    def get_market_details(self, instrument_id: str) -> Instrument:
        pass

    @abstractmethod
    def get_order_book(self, instrument_id: str) -> OrderBook:
        pass

    @abstractmethod
    def get_trade_history(self, instrument_id: str) -> Sequence[Trade]:
        pass

    @abstractmethod
    def get_bars(self, instrument_id: str, count: int = 100) -> Sequence[BarData]:
        pass

    def get_market_data(self, instrument_id: str) -> MarketData:
        details = self.get_market_details(instrument_id)
        bars = list(self.get_bars(instrument_id))
        try:
            ob = self.get_order_book(instrument_id)
        except Exception:
            ob = None
        try:
            trades = list(self.get_trade_history(instrument_id))
        except Exception:
            trades = None
        has_ob = ob is not None and (
            len(getattr(ob, "bids", [])) > 0 or len(getattr(ob, "asks", [])) > 0
        )
        has_trades = trades is not None and len(trades) > 0
        return MarketData(
            instrument_id=instrument_id,
            details=details,
            recent_bars=bars,
            order_book=ob if has_ob else None,
            recent_trades=trades if has_trades else None,
        )
