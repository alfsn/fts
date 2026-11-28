import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict
from quant_core.enums import BarType, Currency, RateType
from quant_core.intervals import interval_to_timedelta
from quant_core.models import BarData as BarDataSchema
from quant_core.models import CorporateAction
from quant_core.models import Instrument as MarketDetailsSchema
from sqlalchemy import func
from sqlalchemy.orm import Session

from .models import BarDataLog as BarDataLogModel
from .models import CorporateActionRecord, FXRateRecord
from .models import Market as MarketModel

logger = logging.getLogger(__name__)


def _naive_utc(ts: datetime) -> datetime:
    """SQLite stores no tz: every timestamp is kept as naive UTC."""
    return ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def _closed(bar: BarDataSchema, now: datetime) -> bool:
    """A time bar is final once its interval has passed (bars are stamped at
    their open). Volume/dollar bars are emitted complete."""
    if bar.bar_type != BarType.TIME or not bar.interval:
        return True
    return _as_utc(bar.timestamp) + interval_to_timedelta(bar.interval) <= now


class CachedSeries(BaseModel):
    """One cached bar series: a symbol at one interval, from one vendor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str
    interval: str
    source: str
    bars: int
    first: datetime
    last: datetime


class BaseRepository:
    def __init__(self, db: Session) -> None:
        self.db = db


class MarketDataRepository(BaseRepository):
    def ensure_market(self, details: MarketDetailsSchema) -> MarketModel:
        try:
            market = (
                self.db.query(MarketModel)
                .filter_by(instrument_id=details.instrument_id)
                .first()
            )
            if not market:
                market = MarketModel(
                    instrument_id=details.instrument_id,
                    name=details.name,
                )
                self.db.add(market)
            else:
                market.name = details.name
            return market
        except Exception as e:
            logger.error(f"Failed to ensure market {details.instrument_id}: {e}")
            raise e

    # --- bars ------------------------------------------------------------------

    def bar_source(self, instrument_id: str, interval: Optional[str]) -> Optional[str]:
        """The vendor the cached series came from, or None if nothing is cached."""
        sources = [
            row[0]
            for row in self.db.query(BarDataLogModel.source)
            .filter_by(instrument_id=instrument_id, interval=interval)
            .distinct()
        ]
        if len(sources) > 1:  # impossible after the PR 2a reset; loud if it happens
            logger.error(f"{instrument_id} {interval} bars mix sources {sources}")
        return sources[0] if sources else None

    def save_bars(
        self, instrument_id: str, bars: Sequence[BarDataSchema], source: str
    ) -> int:
        """
        Upsert bars from ``source``: new timestamps are inserted, existing ones
        updated (late vendor corrections get in). Returns the number inserted.
        Refuses to add to a series cached from another source (D4): switch
        with ``replace_bars`` instead, never stitch.
        A bar still in progress (today's daily bar while the venue trades) is
        never cached: both apps read this cache, and a reader asking for bars
        up to today 00:00 would take it for a close.
        """
        now = datetime.now(timezone.utc)
        partial = [bar for bar in bars if not _closed(bar, now)]
        if partial:
            logger.info(
                f"{instrument_id}: not caching {len(partial)} unfinished bar(s) "
                f"from {partial[0].timestamp}."
            )
            bars = [bar for bar in bars if _closed(bar, now)]
        if not bars:
            return 0
        try:
            bar_type = bars[0].bar_type
            interval = bars[0].interval
            cached_source = self.bar_source(instrument_id, interval)
            if cached_source is not None and cached_source != source:
                raise ValueError(
                    f"{instrument_id} {interval} bars are cached from {cached_source}; "
                    f"refusing to mix in {source}. Use replace_bars."
                )

            existing = {
                row.timestamp: row
                for row in self.db.query(BarDataLogModel).filter_by(
                    instrument_id=instrument_id, bar_type=bar_type, interval=interval
                )
            }
            inserted = 0
            for bar in bars:
                ts = _naive_utc(bar.timestamp)
                values = dict(
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    ticks_count=bar.ticks_count,
                    dollar_volume=bar.dollar_volume,
                    source=source,
                )
                row = existing.get(ts)
                if row is None:
                    row = BarDataLogModel(
                        instrument_id=instrument_id,
                        timestamp=ts,
                        bar_type=bar.bar_type,
                        interval=bar.interval,
                        **values,
                    )
                    self.db.add(row)
                    existing[ts] = row
                    inserted += 1
                else:
                    for key, value in values.items():
                        setattr(row, key, value)
            self.db.flush()
            if inserted:
                logger.info(f"Saved {inserted} new bars for {instrument_id} to SQL.")
            return inserted
        except Exception as e:
            logger.error(f"Failed to save bars for {instrument_id}: {e}")
            raise e

    def replace_bars(
        self,
        instrument_id: str,
        interval: Optional[str],
        bars: Sequence[BarDataSchema],
        source: str,
    ) -> int:
        """Drop the symbol's cached series for ``interval`` and store ``bars``."""
        self.db.query(BarDataLogModel).filter_by(
            instrument_id=instrument_id, interval=interval
        ).delete(synchronize_session=False)
        self.db.flush()
        return self.save_bars(instrument_id, bars, source)

    def _bar_rows(
        self,
        instrument_id: str,
        interval: str,
        start: Optional[datetime],
        end: Optional[datetime],
        bar_type: BarType,
    ):
        query = self.db.query(BarDataLogModel).filter(
            BarDataLogModel.instrument_id == instrument_id,
            BarDataLogModel.interval == interval,
            BarDataLogModel.bar_type == bar_type,
        )
        # SQLite compares the stored naive-UTC text: bounds must be naive UTC too.
        if start is not None:
            query = query.filter(BarDataLogModel.timestamp >= _naive_utc(start))
        if end is not None:
            query = query.filter(BarDataLogModel.timestamp <= _naive_utc(end))
        return query

    @staticmethod
    def _to_bars(rows) -> List[BarDataSchema]:
        """The one row -> ``BarData`` mapping; timestamps come back UTC-aware."""
        bars = [BarDataSchema.model_validate(r, from_attributes=True) for r in rows]
        for bar in bars:
            bar.timestamp = _as_utc(bar.timestamp)
        return bars

    def load_bars(
        self,
        instrument_id: str,
        interval: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        bar_type: BarType = BarType.TIME,
    ) -> List[BarDataSchema]:
        """
        The cached bars of one series, oldest first, raw (as traded). Both
        bounds are inclusive. ``interval`` is required: a symbol cached at two
        intervals is two series, never one.
        """
        rows = self._bar_rows(instrument_id, interval, start, end, bar_type)
        return self._to_bars(rows.order_by(BarDataLogModel.timestamp.asc()))

    def load_bars_before(
        self,
        instrument_id: str,
        interval: str,
        before: datetime,
        count: int,
        bar_type: BarType = BarType.TIME,
    ) -> List[BarDataSchema]:
        """The last ``count`` bars strictly before ``before``, oldest first."""
        rows = (
            self._bar_rows(instrument_id, interval, None, None, bar_type)
            .filter(BarDataLogModel.timestamp < _naive_utc(before))
            .order_by(BarDataLogModel.timestamp.desc())
            .limit(count)
        )
        return self._to_bars(reversed(rows.all()))

    def intervals(self, instrument_id: str) -> List[str]:
        """The intervals cached for a symbol, sorted."""
        rows = (
            self.db.query(BarDataLogModel.interval)
            .filter(BarDataLogModel.instrument_id == instrument_id)
            .distinct()
        )
        return sorted(row[0] for row in rows if row[0] is not None)

    def instruments(self) -> List[str]:
        """Every symbol with cached bars, sorted."""
        rows = self.db.query(BarDataLogModel.instrument_id).distinct()
        return sorted(row[0] for row in rows)

    def series_summary(self) -> List[CachedSeries]:
        """Every cached time-bar series (symbol, interval, source) with its bar
        count and first/last timestamp (UTC), sorted by symbol and interval.
        Volume and dollar bars have no interval and are left out: the
        inspector loads time bars only."""
        rows = (
            self.db.query(
                BarDataLogModel.instrument_id,
                BarDataLogModel.interval,
                BarDataLogModel.source,
                func.count(BarDataLogModel.id),
                func.min(BarDataLogModel.timestamp),
                func.max(BarDataLogModel.timestamp),
            )
            .filter(BarDataLogModel.bar_type == BarType.TIME)
            .group_by(
                BarDataLogModel.instrument_id,
                BarDataLogModel.interval,
                BarDataLogModel.source,
            )
            .order_by(BarDataLogModel.instrument_id, BarDataLogModel.interval)
        )
        return [
            CachedSeries(
                instrument_id=symbol,
                interval=interval,
                source=source,
                bars=bars,
                first=_as_utc(first),
                last=_as_utc(last),
            )
            for symbol, interval, source, bars, first, last in rows
        ]

    def market_name(self, instrument_id: str) -> Optional[str]:
        market = (
            self.db.query(MarketModel).filter_by(instrument_id=instrument_id).first()
        )
        return market.name if market else None

    def counts(self) -> Dict[str, int]:
        """Rows in the cache, for dashboards: markets and bars."""
        return {
            "markets": self.db.query(func.count(MarketModel.id)).scalar() or 0,
            "bars": self.db.query(func.count(BarDataLogModel.id)).scalar() or 0,
        }

    # --- FX --------------------------------------------------------------------

    def get_fx_rate(
        self,
        base_currency: Currency,
        quote_currency: Currency,
        rate_type: RateType,
        as_of: datetime,
    ) -> Optional[float]:
        try:
            record = (
                self.db.query(FXRateRecord)
                .filter(
                    FXRateRecord.base_currency == base_currency,
                    FXRateRecord.quote_currency == quote_currency,
                    FXRateRecord.rate_type == rate_type,
                    FXRateRecord.timestamp <= as_of,
                )
                .order_by(FXRateRecord.timestamp.desc())
                .first()
            )
            return record.rate if record else None
        except Exception as e:
            logger.error(
                f"Failed to get FX rate for {base_currency}/{quote_currency}: {e}"
            )
            raise e

    # --- corporate actions -----------------------------------------------------

    def get_corporate_actions(
        self,
        instrument_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[CorporateActionRecord]:
        try:
            query = self.db.query(CorporateActionRecord).filter(
                CorporateActionRecord.instrument_id == instrument_id
            )
            if start_date:
                query = query.filter(CorporateActionRecord.ex_date >= start_date)
            if end_date:
                query = query.filter(CorporateActionRecord.ex_date <= end_date)

            return query.order_by(CorporateActionRecord.ex_date.asc()).all()
        except Exception as e:
            logger.error(f"Failed to load corporate actions for {instrument_id}: {e}")
            raise e

    def load_corporate_actions(self, instrument_id: str) -> List[CorporateAction]:
        """The cached actions as domain objects, ex-dates in UTC."""
        actions = []
        for row in self.get_corporate_actions(instrument_id):
            action = CorporateAction.model_validate(row, from_attributes=True)
            action.ex_date = _as_utc(action.ex_date)
            actions.append(action)
        return actions

    def save_corporate_actions(
        self, instrument_id: str, actions: Sequence[CorporateAction], source: str
    ) -> None:
        """
        Upsert on (instrument, type, ex-date). Actions cached from another
        source are dropped first, so the cache never mixes vendors (D4).
        """
        existing = {
            (row.action_type, row.ex_date): row
            for row in self.get_corporate_actions(instrument_id)
        }
        if any(row.source != source for row in existing.values()):
            for row in existing.values():
                self.db.delete(row)
            self.db.flush()
            existing = {}
        for action in actions:
            key = (action.action_type, _naive_utc(action.ex_date))
            values = dict(
                record_date=action.record_date and _naive_utc(action.record_date),
                payable_date=action.payable_date and _naive_utc(action.payable_date),
                amount=action.amount,
                ratio=action.ratio,
                source=source,
            )
            row = existing.get(key)
            if row is None:
                self.db.add(
                    CorporateActionRecord(
                        instrument_id=instrument_id,
                        action_type=action.action_type,
                        ex_date=key[1],
                        **values,
                    )
                )
            else:
                for field, value in values.items():
                    setattr(row, field, value)
        self.db.flush()
