# tests/unit/test_backtest_readers.py

import csv
import os
import tempfile
from datetime import datetime, timezone
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from trading_bot.backtesting.abc import BaseBacktestDataReader
from trading_bot.backtesting.readers import CSVBacktestDataReader
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
    assert tick2.market_data["AAPL"].recent_bars[0].close == 101.5


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
