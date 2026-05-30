# src/trading_bot/backtesting/readers.py

import csv
from datetime import datetime
from typing import Iterator

from ..core.enums import BarType
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
