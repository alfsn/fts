# tests/integration/test_data_to_strategy.py

from datetime import datetime, timezone
from typing import Dict, List

import pytest

from trading_bot.core.enums import (
    SignalType,
)

# --- Schemas & Enums (Data Contracts) ---
from trading_bot.core.schemas import (
    ExternalData,
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
    OrderBook,
    PriceLevel,
    Trade,
    TradeSignal,
)

# --- ABCs (Interfaces) ---
from trading_bot.data_ingestion.abc import (
    BaseExternalDataProvider,
    BaseMarketDataProvider,
)

# --- Real Engines to Test ---
from trading_bot.data_ingestion.engine import DataIngestionEngine
from trading_bot.strategy.abc import BaseStrategy
from trading_bot.strategy.engine import StrategyEngine


# --- Fakes (from tests/test_data_ingestion.py) ---
class FakeMarketProvider(BaseMarketDataProvider):
    """
    A fake implementation of the market data provider for testing.
    It returns predictable data.
    """

    def __init__(self, data: Dict[str, MarketData]):
        self.data_to_return = data

    def get_market_data(self, market_id: str) -> MarketData | None:
        """Returns pre-canned data."""
        return self.data_to_return.get(market_id)

    # --- Other ABC methods (not used by the engine) ---
    def list_tradable_markets(self) -> List[MarketDetails]:
        return [md.details for md in self.data_to_return.values()]

    def get_market_details(self, market_id: str) -> MarketDetails:
        return self.data_to_return.get(market_id).details

    def get_order_book(self, market_id: str) -> OrderBook:
        return self.data_to_return.get(market_id).order_book

    def get_trade_history(self, market_id: str) -> List[Trade]:
        return self.data_to_return.get(market_id).recent_trades


class FakeExternalProvider(BaseExternalDataProvider):
    """
    A fake implementation of the external data provider for testing.
    """

    def __init__(self, name: str, data: List[ExternalData]):
        self._name = name
        self.data_to_return = data

    @property
    def source_name(self) -> str:
        """A unique name for the data source."""
        return self._name

    def fetch_data(self) -> List[ExternalData]:
        """Returns pre-canned data."""
        return self.data_to_return


# --- A "Real" Predictable Strategy ---


class SimpleBuyStrategy(BaseStrategy):
    """
    A real, simple strategy that implements the BaseStrategy ABC.
    It generates a BUY signal if the best ask is at or below 0.50.
    """

    @property
    def name(self) -> str:
        return "SimpleTakerStrategy"

    def evaluate(self, data: IngestionEngineOutput) -> List[TradeSignal]:
        signals = []
        # Check the market data for "MKT1"
        market_data = data.market_data.get("MKT1")

        if not market_data:
            return []

        try:
            # Check the best ask price
            best_ask = market_data.order_book.asks[0].price
            if best_ask <= 0.50:
                # Generate a BUY signal
                signals.append(
                    TradeSignal(
                        market_id="MKT1",
                        strategy_name=self.name,
                        signal_type=SignalType.BUY,
                        confidence=1.0,  # Max confidence for this simple rule
                    )
                )
        except IndexError:
            # No asks on the book
            pass

        return signals


# --- Pytest Fixtures ---


@pytest.fixture
def market_data_buy_signal() -> MarketData:
    """A MarketData object that *should* trigger our simple strategy."""
    return MarketData(
        market_id="MKT1",
        order_book=OrderBook(
            bids=[PriceLevel(price=0.49, size=100)],
            asks=[PriceLevel(price=0.50, size=100)],  # Price is <= 0.50
        ),
        recent_trades=[],
        details=MarketDetails(
            market_id="MKT1",
            name="Test Market 1",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        ),
    )


@pytest.fixture
def market_data_hold_signal() -> MarketData:
    """A MarketData object that *should NOT* trigger our simple strategy."""
    return MarketData(
        market_id="MKT1",
        order_book=OrderBook(
            bids=[PriceLevel(price=0.50, size=100)],
            asks=[PriceLevel(price=0.51, size=100)],  # Price is > 0.50
        ),
        recent_trades=[],
        details=MarketDetails(
            market_id="MKT1",
            name="Test Market 1",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        ),
    )


@pytest.fixture
def simple_strategy() -> SimpleBuyStrategy:
    """Returns an instance of our real, simple strategy."""
    return SimpleBuyStrategy()


# --- Integration Test Cases ---


def test_pipeline_generates_buy_signal_on_low_ask(
    market_data_buy_signal, simple_strategy
):
    """
    Tests that the DataIngestionEngine and StrategyEngine work together
    to produce a BUY signal when market conditions are met.
    """
    # --- Arrange ---
    # 1. Set up the Fake provider with the "buy" data
    market_provider = FakeMarketProvider(data={"MKT1": market_data_buy_signal})

    # 2. Instantiate the REAL DataIngestionEngine
    data_engine = DataIngestionEngine(
        market_provider=market_provider,
        external_providers=[],
        market_ids=["MKT1"],
    )

    # 3. Instantiate the REAL StrategyEngine
    strategy_engine = StrategyEngine(strategies=[simple_strategy])

    # --- Act ---
    # 1. Module 1 runs and produces its data contract
    data_packet = data_engine.fetch_all_data()

    # 2. Module 2 runs using the output from Module 1
    signals = strategy_engine.process_data_tick(data_packet)

    # --- Assert ---
    assert len(signals) == 1
    signal = signals[0]
    assert signal.market_id == "MKT1"
    assert signal.signal_type == SignalType.BUY
    assert signal.strategy_name == "SimpleTakerStrategy"


def test_pipeline_generates_no_signal_on_high_ask(
    market_data_hold_signal, simple_strategy
):
    """
    Tests that the DataIngestionEngine and StrategyEngine work together
    to produce no signal when market conditions are not met.
    """
    # --- Arrange ---
    # 1. Set up the Fake provider with the "hold" data
    market_provider = FakeMarketProvider(data={"MKT1": market_data_hold_signal})

    # 2. Instantiate the REAL DataIngestionEngine
    data_engine = DataIngestionEngine(
        market_provider=market_provider,
        external_providers=[],
        market_ids=["MKT1"],
    )

    # 3. Instantiate the REAL StrategyEngine
    strategy_engine = StrategyEngine(strategies=[simple_strategy])

    # --- Act ---
    # 1. Module 1 runs
    data_packet = data_engine.fetch_all_data()

    # 2. Module 2 runs
    signals = strategy_engine.process_data_tick(data_packet)

    # --- Assert ---
    # The strategy's conditions were not met, so no signal should be generated.
    assert len(signals) == 0
