import csv
from datetime import datetime
from typing import Iterator, Optional

from quant_core.enums import AssetType, BarType, Geography
from quant_core.models import TradFiDetails
from sqlalchemy.orm import Session

from ..core.models import BarDataLog as BarDataLogModel
from ..core.models import Market as MarketModel
from ..core.schemas import BarData, IngestionEngineOutput, Instrument, MarketData
from .abc import BaseBacktestDataReader

# src/trading_bot/backtesting/readers.py


class CSVBacktestDataReader(BaseBacktestDataReader):
    """
    Concrete implementation of BaseBacktestDataReader that reads and streams
    historical market data from a local CSV file.
    """

    def __init__(
        self, file_path: str, instrument_id: str, lookback_limit: int = 1000
    ) -> None:
        """
        Initializes the CSV data reader.

        :param file_path: Path to the target CSV file.
        :param instrument_id: The specific market identifier for the stream.
        :param lookback_limit: Maximum number of recent bars to retain in lookback window.
        """
        self.file_path = file_path
        self.instrument_id = instrument_id
        self.lookback_limit = lookback_limit

    def read_data(self) -> Iterator[IngestionEngineOutput]:
        """
        Reads OHLCV bars from the CSV, parses them into Pydantic models,
        and yields sequential IngestionEngineOutput packets in chronological order.
        """
        recent_bars = []
        with open(self.file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Ensure chronological order (fail-safe for sorting)
            rows.sort(key=lambda x: x["timestamp"])

            for row in rows:
                # Convert Z suffix to standard timezone offset if present
                ts_str = row["timestamp"].replace("Z", "+00:00")
                timestamp = datetime.fromisoformat(ts_str)

                # Map row cells to strict BarData fields
                bar = BarData(
                    timestamp=timestamp,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    bar_type=BarType.TIME,
                    ticks_count=int(row.get("ticks_count", 1)),
                    dollar_volume=float(
                        row.get(
                            "dollar_volume", float(row["volume"]) * float(row["close"])
                        )
                    ),
                )

                recent_bars.append(bar)
                if len(recent_bars) > self.lookback_limit:
                    recent_bars.pop(0)

                # Package metadata details
                details = Instrument(
                    instrument_id=self.instrument_id,
                    name=f"{self.instrument_id} Historical Replay",
                    details=TradFiDetails(
                        asset_type=AssetType.STOCK,
                        geography=Geography.US,
                        industry="Tech",
                        currency="USD",
                    ),
                )

                market_data = MarketData(
                    instrument_id=self.instrument_id,
                    details=details,
                    recent_bars=list(recent_bars),
                    order_book=None,
                    recent_trades=None,
                )

                yield IngestionEngineOutput(
                    timestamp=timestamp,
                    market_data={self.instrument_id: market_data},
                    external_data=[],
                    bars={self.instrument_id: list(recent_bars)},
                )


class SQLBacktestDataReader(BaseBacktestDataReader):
    """
    Concrete implementation of BaseBacktestDataReader that reads and streams
    historical market data from a SQLite database via SQLAlchemy.
    """

    def __init__(
        self,
        session: Session,
        instrument_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        warmup_bars: int = 100,
        lookback_limit: int = 1000,
    ) -> None:
        """
        Initializes the SQL data reader.

        :param session: The SQLAlchemy Session to query the database.
        :param instrument_id: The specific market identifier for the stream.
        :param start_date: Optional start datetime (inclusive).
        :param end_date: Optional end datetime (inclusive).
        :param warmup_bars: Number of prior historical bars to query for strategy warm-up.
        :param lookback_limit: Maximum number of recent bars to retain in lookback window.
        """
        self.session = session
        self.instrument_id = instrument_id
        self.start_date = start_date
        self.end_date = end_date
        self.warmup_bars = warmup_bars
        self.lookback_limit = lookback_limit

    def read_data(self) -> Iterator[IngestionEngineOutput]:
        """
        Queries historical bars from the database, parses them into Pydantic models,
        and yields sequential IngestionEngineOutput packets in chronological order.
        """
        from datetime import timezone

        # Resolve market details from DB if available
        market = (
            self.session.query(MarketModel)
            .filter(MarketModel.instrument_id == self.instrument_id)
            .first()
        )
        if market:
            # Handle naive datetime
            end_date = market.end_date
            if end_date and end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            details = Instrument(
                instrument_id=self.instrument_id,
                name=market.name,
                details=TradFiDetails(
                    asset_type=AssetType.STOCK,
                    geography=Geography.US,
                    industry="Tech",
                    currency="USD",
                ),
            )
        else:
            details = Instrument(
                instrument_id=self.instrument_id,
                name=f"{self.instrument_id} Historical Replay",
                details=TradFiDetails(
                    asset_type=AssetType.STOCK,
                    geography=Geography.US,
                    industry="Tech",
                    currency="USD",
                ),
            )

        # 1. Warm-up lookback pre-population
        recent_bars = []
        if self.warmup_bars > 0 and self.start_date:
            prior_bar_logs = (
                self.session.query(BarDataLogModel)
                .filter(
                    BarDataLogModel.instrument_id == self.instrument_id,
                    BarDataLogModel.timestamp < self.start_date,
                )
                .order_by(BarDataLogModel.timestamp.desc())
                .limit(self.warmup_bars)
                .all()
            )
            prior_bar_logs.reverse()

            for db_bar in prior_bar_logs:
                timestamp = db_bar.timestamp
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)

                recent_bars.append(
                    BarData(
                        timestamp=timestamp,
                        open=db_bar.open,
                        high=db_bar.high,
                        low=db_bar.low,
                        close=db_bar.close,
                        volume=db_bar.volume,
                        bar_type=db_bar.bar_type,
                        ticks_count=db_bar.ticks_count,
                        dollar_volume=db_bar.dollar_volume,
                    )
                )

        query = self.session.query(BarDataLogModel).filter(
            BarDataLogModel.instrument_id == self.instrument_id
        )

        if self.start_date:
            query = query.filter(BarDataLogModel.timestamp >= self.start_date)
        if self.end_date:
            query = query.filter(BarDataLogModel.timestamp <= self.end_date)

        # Ensure chronological order
        bars = query.order_by(BarDataLogModel.timestamp.asc()).all()

        for db_bar in bars:
            timestamp = db_bar.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            bar = BarData(
                timestamp=timestamp,
                open=db_bar.open,
                high=db_bar.high,
                low=db_bar.low,
                close=db_bar.close,
                volume=db_bar.volume,
                bar_type=db_bar.bar_type,
                ticks_count=db_bar.ticks_count,
                dollar_volume=db_bar.dollar_volume,
            )

            recent_bars.append(bar)
            if len(recent_bars) > self.lookback_limit:
                recent_bars.pop(0)

            market_data = MarketData(
                instrument_id=self.instrument_id,
                details=details,
                recent_bars=list(recent_bars),
                order_book=None,
                recent_trades=None,
            )

            yield IngestionEngineOutput(
                timestamp=timestamp,
                market_data={self.instrument_id: market_data},
                external_data=[],
                bars={self.instrument_id: list(recent_bars)},
            )
