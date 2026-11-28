from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from .intervals import GapRule, SessionGap
from .models import (
    BarData,
    CorporateAction,
    Instrument,
    MarketData,
    MarketDataRequest,
    OrderBook,
    RateLimit,
    Trade,
)


class RateLimitedError(Exception):
    """The vendor throttled us (HTTP 429 or equivalent). Adapters translate
    their library's error into this one so the throttle can retry it."""


class BaseMarketDataProvider(ABC):
    # Declared request budget; None means the provider needs no throttling
    # (mocks, local files) or throttles itself (ccxt).
    rate_limit: Optional[RateLimit] = None
    # When a hole in this vendor's cached bars is a gap (M2-D5). Exchange
    # hours by default; a 24/7 venue sets ``BarGap()``.
    gap_rule: GapRule = SessionGap()

    @classmethod
    def from_args(cls, args: Any) -> "BaseMarketDataProvider":
        raise NotImplementedError()

    @property
    def source_name(self) -> str:
        """Identifies the vendor in cached bars; one source per symbol."""
        return type(self).__name__

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
        """
        Bars with prices and volumes **as traded (raw), for every interval**.
        A vendor that adjusts history retroactively must undo that before
        returning (``quant_core.analytics.unadjust_splits``). Callers derive
        split-adjusted or total-return series with ``quant_core.analytics.adjust``.
        """
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


class CorporateActionProvider(ABC):
    """Splits and dividends. Separate from market data: it means nothing for crypto."""

    @abstractmethod
    def get_corporate_actions(self, instrument_id: str) -> Sequence[CorporateAction]:
        """
        Full history, raw (as-paid) amounts, ratio = new shares per old share.
        No date range on purpose: un-adjusting a bar needs every split *after*
        it, however far outside the requested range.
        """
        pass
