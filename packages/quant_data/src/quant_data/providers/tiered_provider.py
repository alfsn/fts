import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence, Tuple

from quant_core.interfaces import BaseMarketDataProvider, CorporateActionProvider
from quant_core.intervals import find_gaps
from quant_core.models import (
    BarData,
    CorporateAction,
    Instrument,
    MarketDataRequest,
    OrderBook,
    Trade,
)
from quant_data.db.repositories import MarketDataRepository
from sqlalchemy.orm import Session

from .throttle import Throttle

logger = logging.getLogger(__name__)


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


class TieredDataProvider(BaseMarketDataProvider, CorporateActionProvider):
    """
    Implements a resilient, multi-provider waterfall using the Chain of Responsibility pattern.
    Acts as a drop-in replacement for any standard BaseMarketDataProvider.

    Cache rules (PR 2a): every bar records its ``source``; a symbol's series
    comes from one vendor (a switch refetches the full range and replaces, it
    never stitches); interior gaps trigger one refetch; revised bars are
    updated; corporate actions come from the same vendor as the bars.
    """

    @classmethod
    def from_args(cls, args: Any) -> "TieredDataProvider":
        raise NotImplementedError(
            "TieredDataProvider must be initialized with a db_session and provider_chain."
        )

    def __init__(
        self,
        db_session: Session,
        provider_chain: Sequence[BaseMarketDataProvider],
        throttle: Optional[Throttle] = None,
    ):
        self.repo = MarketDataRepository(db_session)
        self.provider_chain = provider_chain
        # Every vendor call goes through here, so a refetch can't burst (D11).
        self.throttle = throttle or Throttle()

    # --- cache helpers -----------------------------------------------------------

    def _cached_bars(
        self, request: MarketDataRequest, start: datetime, end: datetime
    ) -> list[BarData]:
        return self.repo.load_bars(request.instrument_id, request.interval, start, end)

    def _max_gap(
        self, request: MarketDataRequest, cached_source: Optional[str]
    ) -> timedelta:
        """
        Longest hole the cached series may have before it is refetched: between
        the requested start and the first bar, and between consecutive bars
        (PR 2a, D9). The vendor that filled the cache knows its venue's hours
        (M2-D5); with nothing cached, the first tier decides.
        """
        owner = next(
            (p for p in self.provider_chain if p.source_name == cached_source),
            self.provider_chain[0] if self.provider_chain else BaseMarketDataProvider,
        )
        return owner.gap_rule.max_gap(request.interval)

    @staticmethod
    def _covers_start(
        cached: Sequence[BarData], start: datetime, max_gap: timedelta
    ) -> bool:
        """True if the cache reaches back to `start` (within one allowed gap)."""
        return cached[0].timestamp <= _as_utc(start) + max_gap

    @staticmethod
    def _largest_gap(
        cached: Sequence[BarData], max_gap: timedelta
    ) -> Optional[Tuple[datetime, datetime]]:
        """The first pair of consecutive bars more than ``max_gap`` apart."""
        gaps = find_gaps((b.timestamp for b in cached), max_gap)
        if gaps.empty:
            return None
        first = gaps.iloc[0]
        return first["start"].to_pydatetime(), first["end"].to_pydatetime()

    def _store(
        self,
        request: MarketDataRequest,
        provider: BaseMarketDataProvider,
        data: Sequence[BarData],
        cached_source: Optional[str],
    ) -> None:
        """Upsert from the same vendor; a vendor switch replaces the series (D4)."""
        if cached_source is not None and cached_source != provider.source_name:
            logger.info(
                f"{request.instrument_id}: switching source {cached_source} -> "
                f"{provider.source_name}; replacing the cached series."
            )
            self.repo.replace_bars(
                request.instrument_id, request.interval, data, provider.source_name
            )
        else:
            self.repo.save_bars(request.instrument_id, data, provider.source_name)
        self.repo.db.commit()

    # --- waterfall ---------------------------------------------------------------

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
        A missing start or an interior gap (D9) refetches the full range once.
        Count requests (no `start_date`) keep the "last `count` bars" semantics.
        """
        start, end = request.resolve_bounds()
        cached = self._cached_bars(request, start, end)
        cached_source = self.repo.bar_source(request.instrument_id, request.interval)

        if request.start_date is None:
            if len(cached) >= request.count:
                logger.info(f"Serving {request.instrument_id} bars from DB Cache.")
                return cached[-request.count :]
            data = self._fetch_and_cache(request, cached_source)
            if not data:
                raise self._all_failed(request)
            return data

        max_gap = self._max_gap(request, cached_source)
        gap = self._largest_gap(cached, max_gap) if cached else None
        if cached and self._covers_start(cached, start, max_gap) and gap is None:
            return self._top_up(request, start, end, cached, cached_source)

        if gap is not None:
            logger.info(
                f"{request.instrument_id}: gap {gap[0].date()} -> {gap[1].date()} "
                "in the cache; refetching the full range."
            )
        if not self._fetch_and_cache(request, cached_source):
            if not cached:
                raise self._all_failed(request)
            return cached
        cached = self._cached_bars(request, start, end)
        gap = self._largest_gap(cached, max_gap)
        if gap is not None:
            logger.warning(
                f"{request.instrument_id}: gap {gap[0].date()} -> {gap[1].date()} "
                "persists after a refetch; serving what the vendor has."
            )
        return cached

    def _top_up(
        self,
        request: MarketDataRequest,
        start: datetime,
        end: datetime,
        cached: list[BarData],
        cached_source: Optional[str],
    ) -> Sequence[BarData]:
        """Fetch from the last cached bar on (so a revision to it gets in)."""
        tail = request.model_copy(update={"start_date": cached[-1].timestamp})
        provider, data = self._fetch_from_chain(tail)
        if not data:
            logger.info(f"Serving {request.instrument_id} bars from DB Cache.")
            return cached
        if provider.source_name != cached_source:
            # D4: never stitch two vendors. Refetch everything from the new one.
            try:
                data = self.throttle.call(provider, provider.get_bars, request)
            except Exception as e:
                logger.warning(
                    f"{provider.source_name} failed the full refetch of "
                    f"{request.instrument_id}: {e}; serving the {cached_source} cache."
                )
                return cached
            if not data:
                return cached
        self._store(request, provider, data, cached_source)
        return self._cached_bars(request, start, end)

    def _fetch_and_cache(
        self, request: MarketDataRequest, cached_source: Optional[str]
    ) -> Sequence[BarData]:
        provider, data = self._fetch_from_chain(request)
        if data:
            self._store(request, provider, data, cached_source)
        return data

    @staticmethod
    def _all_failed(request: MarketDataRequest) -> ValueError:
        return ValueError(
            f"All providers in the waterfall failed to get bars for {request.instrument_id}"
        )

    def _fetch_from_chain(
        self, request: MarketDataRequest
    ) -> Tuple[Optional[BaseMarketDataProvider], Sequence[BarData]]:
        for provider in self.provider_chain:
            try:
                data = self.throttle.call(provider, provider.get_bars, request)
                if data:
                    logger.info(
                        f"Serving {request.instrument_id} bars from {provider.__class__.__name__}."
                    )
                    return provider, data
            except Exception as e:
                logger.warning(
                    f"Provider {provider.__class__.__name__} failed get_bars for {request.instrument_id}: {e}"
                )
        return None, []

    # --- corporate actions (D8) ----------------------------------------------------

    def _action_providers(self, instrument_id: str) -> list[CorporateActionProvider]:
        """The vendor the bars came from, or every capable tier if none are cached."""
        capable = [
            p for p in self.provider_chain if isinstance(p, CorporateActionProvider)
        ]
        cached_sources = {
            self.repo.bar_source(instrument_id, interval)
            for interval in self.repo.intervals(instrument_id)
        }
        if cached_sources:
            return [p for p in capable if p.source_name in cached_sources]
        return capable

    def get_corporate_actions(self, instrument_id: str) -> Sequence[CorporateAction]:
        """Refreshed from the vendor on every run; the cache answers when it fails."""
        for provider in self._action_providers(instrument_id):
            try:
                actions = list(
                    self.throttle.call(
                        provider, provider.get_corporate_actions, instrument_id
                    )
                )
            except Exception as e:
                logger.warning(
                    f"{provider.source_name} failed get_corporate_actions for "
                    f"{instrument_id}: {e}; serving the cache."
                )
                continue
            self.repo.save_corporate_actions(
                instrument_id, actions, provider.source_name
            )
            self.repo.db.commit()
            return actions
        return self.repo.load_corporate_actions(instrument_id)
