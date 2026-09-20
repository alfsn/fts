# tests/test_strategy.py

import logging
from datetime import datetime, timezone
from typing import List

import pytest

# Import ABCs and Schemas
from trading_bot.core.schemas import (
    ExternalData,
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
    OrderBook,
    SignalType,
    TradeSignal,
)
from trading_bot.strategy.abc import BaseStrategy

# Import the class we are testing
from trading_bot.strategy.engine import StrategyEngine

# --- Fake (Stub) Implementations for Testing ---


class FakeStrategy(BaseStrategy):
    """
    A fake implementation of a strategy for testing.
    It can be configured to return specific signals or to fail.
    """

    def __init__(
        self,
        strategy_name: str,
        signals_to_return: List[TradeSignal],
        should_fail: bool = False,
        wrong_name: bool = False,
    ):
        self._name = strategy_name
        self.signals_to_return = signals_to_return
        self.should_fail = should_fail
        self.wrong_name = wrong_name

    @property
    def name(self) -> str:
        """The strategy's name."""
        return self._name

    def evaluate(self, data: IngestionEngineOutput) -> List[TradeSignal]:
        """Returns pre-canned signals or simulates a failure."""
        if self.should_fail:
            raise ValueError(f"Simulated failure for {self.name}")

        if self.wrong_name:
            # Return signals but with an incorrect name
            wrong_signals = [
                s.model_copy(update={"strategy_name": "WRONG_NAME"})
                for s in self.signals_to_return
            ]
            return wrong_signals

        return self.signals_to_return


# --- Pytest Fixtures ---


@pytest.fixture
def mock_signal_1() -> TradeSignal:
    """A sample signal from 'strat_A'."""
    return TradeSignal(
        market_id="MKT1",
        strategy_name="strat_A",
        signal_type=SignalType.BUY,
        confidence=0.7,
    )


@pytest.fixture
def mock_signal_2() -> TradeSignal:
    """A sample signal from 'strat_B'."""
    return TradeSignal(
        market_id="MKT2",
        strategy_name="strat_B",
        signal_type=SignalType.SELL,
        confidence=0.8,
    )


@pytest.fixture
def mock_data_tick() -> IngestionEngineOutput:
    """A sample IngestionEngineOutput data packet."""
    market_data = MarketData(
        market_id="MKT1",
        order_book=OrderBook(bids=[], asks=[]),
        recent_trades=[],
        details=MarketDetails(
            market_id="MKT1",
            name="Test",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        ),
    )
    external_data = ExternalData(
        source="test",
        timestamp=datetime.now(timezone.utc),
        content={},
    )
    return IngestionEngineOutput(
        timestamp=datetime.now(timezone.utc),
        market_data={"MKT1": market_data},
        external_data=[external_data],
    )


# --- Test Cases ---


def test_engine_initialization(mock_signal_1):
    """Tests that the engine correctly stores its injected strategies."""
    # Given
    strategy_a = FakeStrategy(
        strategy_name="strat_A", signals_to_return=[mock_signal_1]
    )
    strategies = [strategy_a]

    # When
    engine = StrategyEngine(strategies=strategies)

    # Then
    assert engine.strategies == strategies
    assert len(engine.strategies) == 1


def test_engine_no_strategies(mock_data_tick, caplog):
    """Tests that the engine runs and returns an empty list if no
    strategies are provided."""
    # Given
    engine = StrategyEngine(strategies=[])

    # When
    with caplog.at_level(logging.WARNING):
        signals = engine.process_data_tick(mock_data_tick)

    # Then
    assert signals == []
    assert "StrategyEngine initialized with zero strategies" in caplog.text


def test_engine_one_strategy_happy_path(mock_data_tick, mock_signal_1):
    """
    Tests the engine's main loop with one strategy, expecting
    a successful signal list.
    """
    # Given
    strategy_a = FakeStrategy(
        strategy_name="strat_A", signals_to_return=[mock_signal_1]
    )
    engine = StrategyEngine(strategies=[strategy_a])

    # When
    signals = engine.process_data_tick(mock_data_tick)

    # Then
    assert len(signals) == 1
    assert signals[0] == mock_signal_1
    assert signals[0].strategy_name == "strat_A"


def test_engine_multiple_strategies_happy_path(
    mock_data_tick, mock_signal_1, mock_signal_2
):
    """
    Tests that the engine correctly polls and assembles signals from
    multiple strategies.
    """
    # Given
    strategy_a = FakeStrategy(
        strategy_name="strat_A", signals_to_return=[mock_signal_1]
    )
    strategy_b = FakeStrategy(
        strategy_name="strat_B", signals_to_return=[mock_signal_2]
    )
    engine = StrategyEngine(strategies=[strategy_a, strategy_b])

    # When
    signals = engine.process_data_tick(mock_data_tick)

    # Then
    assert len(signals) == 2
    assert mock_signal_1 in signals
    assert mock_signal_2 in signals


def test_engine_strategy_failure_isolation(mock_data_tick, mock_signal_1, caplog):
    """
    Tests that the engine is robust to a failure in *one* of the
    strategies. It should log the error but still process the others.
    """
    # Given
    strategy_a_success = FakeStrategy(
        strategy_name="strat_A", signals_to_return=[mock_signal_1]
    )
    strategy_b_fail = FakeStrategy(
        strategy_name="strat_B", signals_to_return=[], should_fail=True
    )
    engine = StrategyEngine(strategies=[strategy_a_success, strategy_b_fail])

    # When
    with caplog.at_level(logging.ERROR):
        signals = engine.process_data_tick(mock_data_tick)

    # Then
    # Check that we still got the signal from the good strategy
    assert len(signals) == 1
    assert signals[0] == mock_signal_1

    # Check that the failure was logged
    assert "Error during evaluation of strategy 'strat_B'" in caplog.text
    assert "Simulated failure for strat_B" in caplog.text


def test_engine_signal_name_correction(mock_data_tick, mock_signal_1, caplog):
    """
    Tests that the engine corrects a signal's strategy_name if it
    doesn't match the strategy that produced it.
    """
    # Given
    strategy_a = FakeStrategy(
        strategy_name="strat_A",
        signals_to_return=[mock_signal_1],
        wrong_name=True,
    )
    engine = StrategyEngine(strategies=[strategy_a])

    # When
    with caplog.at_level(logging.WARNING):
        signals = engine.process_data_tick(mock_data_tick)

    # Then
    # Check that the signal's name was corrected
    assert len(signals) == 1
    assert signals[0].strategy_name == "strat_A"
    assert signals[0].market_id == "MKT1"

    # Check that a warning was logged
    assert "has mismatched name: 'WRONG_NAME'. Correcting." in caplog.text
