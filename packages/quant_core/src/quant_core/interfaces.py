from abc import ABC, abstractmethod
from typing import Any, Sequence

from .models import BarData, Instrument, MarketData, MarketDataRequest, OrderBook, Trade


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
    def get_bars(self, request: MarketDataRequest) -> Sequence[BarData]:
        pass

    def get_market_data(self, request: MarketDataRequest) -> MarketData:
        details = self.get_market_details(request.instrument_id)
        bars = list(self.get_bars(request))

        ob = None
        if request.include_order_book:
            try:
                ob = self.get_order_book(request.instrument_id)
            except Exception:
                pass

        trades = None
        if request.include_trades:
            try:
                trades = list(self.get_trade_history(request.instrument_id))
            except Exception:
                pass

        has_ob = ob is not None and (
            len(getattr(ob, "bids", [])) > 0 or len(getattr(ob, "asks", [])) > 0
        )
        has_trades = trades is not None and len(trades) > 0
        return MarketData(
            instrument_id=request.instrument_id,
            details=details,
            recent_bars=bars,
            order_book=ob if has_ob else None,
            recent_trades=trades if has_trades else None,
        )
