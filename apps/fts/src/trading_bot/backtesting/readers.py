import csv
from datetime import datetime
from typing import Iterator, Optional

from quant_core.enums import AssetType, BarType, Geography
from quant_core.models import TradFiDetails
from quant_data.db import MarketDataRepository
from sqlalchemy.orm import Session

from ..core.bars import load_research_bars
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
                    market_data=[market_data],
                    external_data=[],
                    bars={self.instrument_id: list(recent_bars)},
                )


class SQLBacktestDataReader(BaseBacktestDataReader):
    """
    Streams one cached bar series (instrument + interval) from the price
    cache. The cache is read only through ``quant_data``'s repository (M2a);
    bars are raw and a split inside the window raises (``core.bars``).
    """

    def __init__(
        self,
        instrument_id: str,
        interval: str,
        session: Optional[Session] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        warmup_bars: int = 100,
        lookback_limit: int = 1000,
    ) -> None:
        """
        :param instrument_id: The specific market identifier for the stream.
        :param interval: The bar interval (e.g. '1h', '1d'); required, so a
            symbol cached at two intervals never mixes them.
        :param session: Session on the price cache.
        :param start_date: Optional start datetime (inclusive).
        :param end_date: Optional end datetime (inclusive).
        :param warmup_bars: Number of prior historical bars to query for strategy warm-up.
        :param lookback_limit: Maximum number of recent bars to retain in lookback window.
        """
        if session is None:
            from trading_bot.core.database import SessionLocal

            session = SessionLocal()
        self.repo = MarketDataRepository(session)
        self.instrument_id = instrument_id
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.warmup_bars = warmup_bars
        self.lookback_limit = lookback_limit

    def read_data(self) -> Iterator[IngestionEngineOutput]:
        """
        Yields one IngestionEngineOutput per bar, oldest first, each with the
        rolling window of the last ``lookback_limit`` bars (warm-up included).
        """
        name = self.repo.market_name(self.instrument_id)
        details = Instrument(
            instrument_id=self.instrument_id,
            name=name or f"{self.instrument_id} Historical Replay",
            details=TradFiDetails(
                asset_type=AssetType.STOCK,
                geography=Geography.US,
                industry="Tech",
                currency="USD",
            ),
        )
        loaded = load_research_bars(
            self.repo,
            self.instrument_id,
            self.interval,
            self.start_date,
            self.end_date,
            warmup=self.warmup_bars,
        )

        recent_bars = list(loaded.warmup)
        for bar in loaded.bars:
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
                timestamp=bar.timestamp,
                market_data=[market_data],
                external_data=[],
                bars={self.instrument_id: list(recent_bars)},
            )
