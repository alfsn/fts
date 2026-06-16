# src/trading_bot/backtesting/readers.py

import csv
from datetime import datetime
from typing import Iterator, Optional

from sqlalchemy.orm import Session

from ..core.enums import BarType
from ..core.models import BarDataLog as BarDataLogModel
from ..core.models import Market as MarketModel
from ..core.schemas import BarData, IngestionEngineOutput, MarketData, MarketDetails
from .abc import BaseBacktestDataReader


class CSVBacktestDataReader(BaseBacktestDataReader):
    """
    Concrete implementation of BaseBacktestDataReader that reads and streams
    historical market data from a local CSV file.
    """

    def __init__(self, file_path: str, market_id: str) -> None:
        """
        Initializes the CSV data reader.

        :param file_path: Path to the target CSV file.
        :param market_id: The specific market identifier for the stream.
        """
        self.file_path = file_path
        self.market_id = market_id

    def read_data(self) -> Iterator[IngestionEngineOutput]:
        """
        Reads OHLCV bars from the CSV, parses them into Pydantic models,
        and yields sequential IngestionEngineOutput packets in chronological order.
        """
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

                # Package metadata details
                details = MarketDetails(
                    market_id=self.market_id,
                    name=f"{self.market_id} Historical Replay",
                    end_date=timestamp,
                    resolution_source="csv_replay",
                )

                market_data = MarketData(
                    market_id=self.market_id,
                    details=details,
                    recent_bars=[bar],
                    order_book=None,
                    recent_trades=None,
                )

                yield IngestionEngineOutput(
                    timestamp=timestamp,
                    market_data={self.market_id: market_data},
                    external_data=[],
                    bars={self.market_id: [bar]},
                )


class SQLBacktestDataReader(BaseBacktestDataReader):
    """
    Concrete implementation of BaseBacktestDataReader that reads and streams
    historical market data from a SQLite database via SQLAlchemy.
    """

    def __init__(
        self,
        session: Session,
        market_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Initializes the SQL data reader.

        :param session: The SQLAlchemy Session to query the database.
        :param market_id: The specific market identifier for the stream.
        :param start_date: Optional start datetime (inclusive).
        :param end_date: Optional end datetime (inclusive).
        """
        self.session = session
        self.market_id = market_id
        self.start_date = start_date
        self.end_date = end_date

    def read_data(self) -> Iterator[IngestionEngineOutput]:
        """
        Queries historical bars from the database, parses them into Pydantic models,
        and yields sequential IngestionEngineOutput packets in chronological order.
        """
        from datetime import timezone

        # Resolve market details from DB if available
        market = (
            self.session.query(MarketModel)
            .filter(MarketModel.market_id == self.market_id)
            .first()
        )
        if market:
            # Handle naive datetime
            end_date = market.end_date
            if end_date and end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            details = MarketDetails(
                market_id=self.market_id,
                name=market.name,
                end_date=end_date,
                resolution_source=market.resolution_source or "sqlite_replay",
            )
        else:
            details = MarketDetails(
                market_id=self.market_id,
                name=f"{self.market_id} Historical Replay",
                end_date=datetime.now(timezone.utc),
                resolution_source="sqlite_replay",
            )

        query = self.session.query(BarDataLogModel).filter(
            BarDataLogModel.market_id == self.market_id
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

            market_data = MarketData(
                market_id=self.market_id,
                details=details,
                recent_bars=[bar],
                order_book=None,
                recent_trades=None,
            )

            yield IngestionEngineOutput(
                timestamp=timestamp,
                market_data={self.market_id: market_data},
                external_data=[],
                bars={self.market_id: [bar]},
            )
