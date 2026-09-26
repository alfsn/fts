import logging
from datetime import datetime
from typing import List, Optional, Sequence

from quant_core.enums import BarType
from quant_core.models import BarData as BarDataSchema
from quant_core.models import Instrument as MarketDetailsSchema
from sqlalchemy.orm import Session

from .models import BarDataLog as BarDataLogModel
from .models import Market as MarketModel

logger = logging.getLogger(__name__)


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

    def save_bars(self, instrument_id: str, bars: Sequence[BarDataSchema]) -> int:
        if not bars:
            return 0
        try:
            bar_type = bars[0].bar_type
            interval = bars[0].interval
            from datetime import timezone

            existing_records = (
                self.db.query(BarDataLogModel.timestamp)
                .filter_by(
                    instrument_id=instrument_id, bar_type=bar_type, interval=interval
                )
                .all()
            )
            timestamps = {
                (
                    r[0].astimezone(timezone.utc).replace(tzinfo=None)
                    if r[0].tzinfo
                    else r[0]
                )
                for r in existing_records
            }

            new_logs = []
            for bar in bars:
                ts = bar.timestamp
                ts_normalized = (
                    ts.astimezone(timezone.utc).replace(tzinfo=None)
                    if ts.tzinfo
                    else ts
                )
                if ts_normalized not in timestamps:
                    log = BarDataLogModel(
                        instrument_id=instrument_id,
                        timestamp=ts_normalized,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        bar_type=bar.bar_type,
                        interval=bar.interval,
                        ticks_count=bar.ticks_count,
                        dollar_volume=bar.dollar_volume,
                    )
                    new_logs.append(log)
                    timestamps.add(ts_normalized)

            if new_logs:
                self.db.bulk_save_objects(new_logs)
                logger.info(
                    f"Saved {len(new_logs)} new bars for {instrument_id} to SQL."
                )
            return len(new_logs)
        except Exception as e:
            logger.error(f"Failed to save bars for {instrument_id}: {e}")
            raise e

    def get_bars(
        self,
        market_ids: str | Sequence[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bar_type: BarType = BarType.TIME,
        interval: Optional[str] = None,
    ) -> List[BarDataLogModel]:
        try:
            if isinstance(market_ids, str):
                market_ids = [market_ids]

            query = self.db.query(BarDataLogModel).filter(
                BarDataLogModel.instrument_id.in_(market_ids),
                BarDataLogModel.bar_type == bar_type,
            )
            if interval is not None:
                query = query.filter(BarDataLogModel.interval == interval)
            if start_date:
                query = query.filter(BarDataLogModel.timestamp >= start_date)
            if end_date:
                query = query.filter(BarDataLogModel.timestamp <= end_date)

            return query.order_by(BarDataLogModel.timestamp.asc()).all()
        except Exception as e:
            logger.error(f"Failed to load bars for {market_ids}: {e}")
            raise e
