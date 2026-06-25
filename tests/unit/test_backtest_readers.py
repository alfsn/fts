# tests/unit/test_backtest_readers.py

import csv
import os
import tempfile
from datetime import datetime, timezone
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from trading_bot.backtesting.abc import BaseBacktestDataReader
from trading_bot.backtesting.readers import CSVBacktestDataReader, SQLBacktestDataReader
from trading_bot.core.enums import BarType
from trading_bot.core.loop import HistoricalReplayLoop
from trading_bot.core.pipeline import TradingPipeline
from trading_bot.core.schemas import (
    BarData,
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
)


# Define a MockDataReader that inherits from BaseBacktestDataReader
class MockBacktestDataReader(BaseBacktestDataReader):
    """A mock implementation of BaseBacktestDataReader for testing loops."""

    def __init__(self, ticks: list[IngestionEngineOutput]) -> None:
        self.ticks = ticks

    def read_data(self) -> Iterator[IngestionEngineOutput]:
        for tick in self.ticks:
            yield tick


@pytest.fixture
def temp_csv_file():
    """Creates a temporary CSV file with mock historical OHLCV data."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            writer.writerow(
                ["2026-05-30T12:00:00Z", "100.0", "105.0", "99.0", "102.0", "500.0"]
            )
            writer.writerow(
                ["2026-05-30T12:01:00Z", "102.0", "103.0", "101.0", "101.5", "300.0"]
            )
        yield path
    finally:
        os.unlink(path)


def test_csv_backtest_data_reader(temp_csv_file):
    """
    Verifies that CSVBacktestDataReader correctly parses CSV rows
    and yields chronological IngestionEngineOutput packets.
    """
    reader = CSVBacktestDataReader(file_path=temp_csv_file, market_id="AAPL")
    ticks = list(reader.read_data())

    # Assertions
    assert len(ticks) == 2

    # Check first tick
    tick1 = ticks[0]
    assert isinstance(tick1, IngestionEngineOutput)
    assert tick1.timestamp == datetime.fromisoformat("2026-05-30T12:00:00+00:00")
    assert "AAPL" in tick1.market_data

    mdata1 = tick1.market_data["AAPL"]
    assert isinstance(mdata1, MarketData)
    assert mdata1.market_id == "AAPL"
    assert mdata1.details.name == "AAPL Historical Replay"
    assert mdata1.details.resolution_source == "csv_replay"

    bar1 = mdata1.recent_bars[0]
    assert isinstance(bar1, BarData)
    assert bar1.open == 100.0
    assert bar1.high == 105.0
    assert bar1.low == 99.0
    assert bar1.close == 102.0
    assert bar1.volume == 500.0
    assert bar1.bar_type == BarType.TIME

    # Check second tick
    tick2 = ticks[1]
    assert tick2.timestamp == datetime.fromisoformat("2026-05-30T12:01:00+00:00")
    assert tick2.market_data["AAPL"].recent_bars[-1].close == 101.5


def test_historical_replay_loop_dependency_injection():
    """
    Verifies that HistoricalReplayLoop correctly accepts an injected
    BaseBacktestDataReader and streams ticks chronologically to the pipeline.
    """
    # 1. Arrange Mock Data
    mock_details = MarketDetails(
        market_id="MKT-1",
        name="Test",
        end_date=datetime.now(timezone.utc),
        resolution_source="test",
    )
    mock_market_data = MarketData(
        market_id="MKT-1",
        details=mock_details,
        recent_bars=[],
    )
    tick1 = IngestionEngineOutput(
        timestamp=datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc),
        market_data={"MKT-1": mock_market_data},
        external_data=[],
    )
    tick2 = IngestionEngineOutput(
        timestamp=datetime(2026, 5, 30, 12, 1, 0, tzinfo=timezone.utc),
        market_data={"MKT-1": mock_market_data},
        external_data=[],
    )

    mock_reader = MockBacktestDataReader([tick1, tick2])
    mock_pipeline = MagicMock(spec=TradingPipeline)

    # 2. Act
    loop_driver = HistoricalReplayLoop(data_reader=mock_reader)
    loop_driver.start(pipeline=mock_pipeline)

    # 3. Assert
    assert mock_pipeline.execute_single_tick.call_count == 2
    # Ensure sequential chronological order of calls
    calls = mock_pipeline.execute_single_tick.call_args_list
    assert calls[0][1]["ingestion_output"] == tick1
    assert calls[1][1]["ingestion_output"] == tick2


def test_historical_replay_loop_backwards_compatibility_fallback(temp_csv_file):
    """
    Verifies that passing data_path to HistoricalReplayLoop triggers the
    fallback compatibility flow and successfully replays OHLCV bars.
    """
    mock_pipeline = MagicMock(spec=TradingPipeline)

    # Instantiate using legacy data_path signature
    loop_driver = HistoricalReplayLoop(data_path=temp_csv_file)
    loop_driver.start(pipeline=mock_pipeline)

    # Should fall back to CSVBacktestDataReader and process the 2 ticks in the CSV
    assert mock_pipeline.execute_single_tick.call_count == 2

    # Verify first replayed tick parameters
    called_tick = mock_pipeline.execute_single_tick.call_args_list[0][1][
        "ingestion_output"
    ]
    assert isinstance(called_tick, IngestionEngineOutput)
    assert "mock-btc" in called_tick.market_data
    assert called_tick.market_data["mock-btc"].recent_bars[0].close == 102.0


@pytest.fixture
def sql_db_session():
    """Sets up an in-memory SQLite database and creates the schema."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from trading_bot.core.database import Base
    from trading_bot.core.enums import BarType
    from trading_bot.core.models import BarDataLog, Market

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(bind=engine)
    session = SessionClass()

    # Pre-populate a market record since foreign key constraints exist
    market = Market(
        market_id="AAPL",
        name="Apple Inc.",
        end_date=datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        resolution_source="sqlite_test",
    )
    session.add(market)

    # Pre-populate some historical bars
    bar1 = BarDataLog(
        market_id="AAPL",
        timestamp=datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc),
        open=100.0,
        high=105.0,
        low=99.0,
        close=102.0,
        volume=500.0,
        bar_type=BarType.TIME,
        ticks_count=10,
        dollar_volume=51000.0,
    )
    bar2 = BarDataLog(
        market_id="AAPL",
        timestamp=datetime(2026, 5, 30, 12, 1, 0, tzinfo=timezone.utc),
        open=102.0,
        high=103.0,
        low=101.0,
        close=101.5,
        volume=300.0,
        bar_type=BarType.TIME,
        ticks_count=5,
        dollar_volume=30450.0,
    )
    session.add_all([bar1, bar2])
    session.commit()

    try:
        yield session
    finally:
        session.close()


def test_sql_backtest_data_reader(sql_db_session):
    """
    Verifies that SQLBacktestDataReader correctly parses SQL rows
    and yields chronological IngestionEngineOutput packets.
    """
    reader = SQLBacktestDataReader(session=sql_db_session, market_id="AAPL")
    ticks = list(reader.read_data())

    # Assertions
    assert len(ticks) == 2

    # Check first tick
    tick1 = ticks[0]
    assert isinstance(tick1, IngestionEngineOutput)
    assert tick1.timestamp == datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
    assert "AAPL" in tick1.market_data

    mdata1 = tick1.market_data["AAPL"]
    assert isinstance(mdata1, MarketData)
    assert mdata1.market_id == "AAPL"
    assert mdata1.details.name == "Apple Inc."
    assert mdata1.details.resolution_source == "sqlite_test"

    bar1 = mdata1.recent_bars[0]
    assert isinstance(bar1, BarData)
    assert bar1.open == 100.0
    assert bar1.high == 105.0
    assert bar1.low == 99.0
    assert bar1.close == 102.0
    assert bar1.volume == 500.0
    assert bar1.bar_type == BarType.TIME

    # Check second tick
    tick2 = ticks[1]
    assert tick2.timestamp == datetime(2026, 5, 30, 12, 1, 0, tzinfo=timezone.utc)
    assert tick2.market_data["AAPL"].recent_bars[-1].close == 101.5


def test_sql_backtest_data_reader_with_date_filters(sql_db_session):
    """Verifies that SQLBacktestDataReader respects start_date and end_date filters."""
    start_date = datetime(2026, 5, 30, 12, 0, 30, tzinfo=timezone.utc)
    reader = SQLBacktestDataReader(
        session=sql_db_session,
        market_id="AAPL",
        start_date=start_date,
    )
    ticks = list(reader.read_data())
    assert len(ticks) == 1
    assert ticks[0].timestamp == datetime(2026, 5, 30, 12, 1, 0, tzinfo=timezone.utc)

    end_date = datetime(2026, 5, 30, 12, 0, 30, tzinfo=timezone.utc)
    reader_end = SQLBacktestDataReader(
        session=sql_db_session,
        market_id="AAPL",
        end_date=end_date,
    )
    ticks_end = list(reader_end.read_data())
    assert len(ticks_end) == 1
    assert ticks_end[0].timestamp == datetime(
        2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc
    )


def test_sql_backtest_data_reader_fallback_market(sql_db_session):
    """Verifies fallback behavior when market details are not found in the database."""
    from trading_bot.core.models import BarDataLog

    # Add a bar for an unsaved market ID
    bar = BarDataLog(
        market_id="GOOG",
        timestamp=datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc),
        open=200.0,
        high=205.0,
        low=199.0,
        close=202.0,
        volume=100.0,
        bar_type=BarType.TIME,
        ticks_count=10,
        dollar_volume=20200.0,
    )
    sql_db_session.add(bar)
    sql_db_session.commit()

    reader = SQLBacktestDataReader(session=sql_db_session, market_id="GOOG")
    ticks = list(reader.read_data())
    assert len(ticks) == 1
    assert ticks[0].market_data["GOOG"].details.name == "GOOG Historical Replay"
    assert ticks[0].market_data["GOOG"].details.resolution_source == "sqlite_replay"


def test_sql_backtest_data_reader_warmup_and_lookback(sql_db_session):
    """Verifies that SQLBacktestDataReader respects warmup_bars and lookback_limit."""
    from trading_bot.core.enums import BarType
    from trading_bot.core.models import BarDataLog

    # Clear pre-populated logs to have complete control
    sql_db_session.query(BarDataLog).delete()
    sql_db_session.commit()

    # Pre-populate historical bars before start_date (warmup)
    warmup_time1 = datetime(2026, 5, 30, 11, 0, 0, tzinfo=timezone.utc)
    warmup_time2 = datetime(2026, 5, 30, 11, 30, 0, tzinfo=timezone.utc)
    bar_warmup1 = BarDataLog(
        market_id="AAPL",
        timestamp=warmup_time1,
        open=98.0,
        high=99.0,
        low=97.0,
        close=98.5,
        volume=100.0,
        bar_type=BarType.TIME,
        ticks_count=5,
        dollar_volume=9850.0,
    )
    bar_warmup2 = BarDataLog(
        market_id="AAPL",
        timestamp=warmup_time2,
        open=98.5,
        high=100.0,
        low=98.0,
        close=99.5,
        volume=150.0,
        bar_type=BarType.TIME,
        ticks_count=7,
        dollar_volume=14925.0,
    )

    # Pre-populate actual bars during/after start_date
    start_time = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
    actual_time1 = start_time
    actual_time2 = datetime(2026, 5, 30, 12, 1, 0, tzinfo=timezone.utc)
    bar_actual1 = BarDataLog(
        market_id="AAPL",
        timestamp=actual_time1,
        open=100.0,
        high=105.0,
        low=99.0,
        close=102.0,
        volume=500.0,
        bar_type=BarType.TIME,
        ticks_count=10,
        dollar_volume=51000.0,
    )
    bar_actual2 = BarDataLog(
        market_id="AAPL",
        timestamp=actual_time2,
        open=102.0,
        high=103.0,
        low=101.0,
        close=101.5,
        volume=300.0,
        bar_type=BarType.TIME,
        ticks_count=5,
        dollar_volume=30450.0,
    )

    sql_db_session.add_all([bar_warmup1, bar_warmup2, bar_actual1, bar_actual2])
    sql_db_session.commit()

    # Initialize reader with warmup_bars=2 and lookback_limit=3
    reader = SQLBacktestDataReader(
        session=sql_db_session,
        market_id="AAPL",
        start_date=start_time,
        warmup_bars=2,
        lookback_limit=3,
    )

    ticks = list(reader.read_data())

    # We expect exactly 2 ticks corresponding to the actual data (at/after start_date)
    assert len(ticks) == 2

    # Tick 1: should contain: warmup1, warmup2, actual1
    tick1 = ticks[0]
    bars_t1 = tick1.market_data["AAPL"].recent_bars
    assert len(bars_t1) == 3
    assert bars_t1[0].timestamp == warmup_time1
    assert bars_t1[1].timestamp == warmup_time2
    assert bars_t1[2].timestamp == actual_time1

    # Tick 2: should contain: warmup2, actual1, actual2 (warmup1 is dropped due to lookback_limit=3)
    tick2 = ticks[1]
    bars_t2 = tick2.market_data["AAPL"].recent_bars
    assert len(bars_t2) == 3
    assert bars_t2[0].timestamp == warmup_time2
    assert bars_t2[1].timestamp == actual_time1
    assert bars_t2[2].timestamp == actual_time2
