# tests/unit/test_event_loop.py

from unittest.mock import MagicMock

import pytest

from trading_bot.core.loop import HistoricalReplayLoop, RealTimePollingLoop
from trading_bot.core.pipeline import TradingPipeline


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Mocks the TradingPipeline."""
    return MagicMock(spec=TradingPipeline)


def test_real_time_polling_loop_limit(mock_pipeline):
    """
    Tests that RealTimePollingLoop periodically triggers the pipeline tick
    and terminates successfully after reaching the max_ticks limit.
    """
    # Initialize with a very small sleep interval (0.01 seconds) and a limit of 2 ticks
    loop_driver = RealTimePollingLoop(interval_seconds=0.01, max_ticks=2)

    # Start loop
    loop_driver.start(pipeline=mock_pipeline)

    # Assertions: Pipeline was triggered exactly twice
    assert mock_pipeline.execute_single_tick.call_count == 2


def test_historical_replay_loop(mock_pipeline):
    """
    Tests that HistoricalReplayLoop executes the pipeline tick once
    representing the data replay cycle.
    """
    loop_driver = HistoricalReplayLoop(data_path="mock_history.csv")

    # Start loop
    loop_driver.start(pipeline=mock_pipeline)

    # Assertions: Pipeline tick executed once
    assert mock_pipeline.execute_single_tick.call_count == 1
