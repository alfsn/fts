# tests/unit/test_event_loop.py

from unittest.mock import MagicMock

import pytest

from trading_bot.core.loop import HistoricalReplayLoop, RealTimePollingLoop
from trading_bot.core.models import BacktestPredictionLog, ModelPredictionLog
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
    from datetime import datetime, timezone

    from trading_bot.backtesting.abc import BaseBacktestDataReader

    mock_reader = MagicMock(spec=BaseBacktestDataReader)
    mock_tick = MagicMock()
    mock_tick.timestamp = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
    mock_reader.read_data.return_value = [mock_tick]

    loop_driver = HistoricalReplayLoop(data_reader=mock_reader)

    # Start loop
    loop_driver.start(pipeline=mock_pipeline)

    # Assertions: Pipeline tick executed once
    assert mock_pipeline.execute_single_tick.call_count == 1


def test_event_loop_prediction_log_model_property():
    """
    Tests that loop drivers correctly expose the polymorphic prediction_log_model property.
    """
    polling_loop = RealTimePollingLoop(interval_seconds=0.01, max_ticks=1)
    replay_loop = HistoricalReplayLoop()

    assert polling_loop.prediction_log_model is ModelPredictionLog
    assert replay_loop.prediction_log_model is BacktestPredictionLog
