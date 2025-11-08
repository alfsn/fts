# tests/integration/test_strategy_to_risk_flow.py

from datetime import datetime, timezone
from typing import List

import pytest

# Import Schemas
from trading_bot.core.schemas import (
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
    OrderBook,
    PriceLevel,
    SignalType,
    TradeSignal,
)
from trading_bot.risk_management.manager import RiskManager
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.risk_management.sizing.fixed_amount import FixedAmountSizer

# Import ABCs
from trading_bot.strategy.abc import BaseStrategy

# Import Real Components to Test
from trading_bot.strategy.engine import StrategyEngine

# --- A simple "real" strategy for testing ---


class SimpleBuyStrategy(BaseStrategy):
    """A simple strategy that buys if the ask price is below 0.50."""

    @property
    def name(self) -> str:
        return "simple_buy_strat"

    def evaluate(self, data: IngestionEngineOutput) -> List[TradeSignal]:
        signals = []
        market_data = data.market_data.get("MKT-01")

        if not market_data or not market_data.order_book.asks:
            return []

        best_ask = market_data.order_book.asks[0].price

        if best_ask < 0.50:
            signals.append(
                TradeSignal(
                    market_id="MKT-01",
                    strategy_name=self.name,
                    signal_type=SignalType.BUY,
                    outcome="yes",
                    confidence=0.8,
                )
            )
        else:
            signals.append(
                TradeSignal(
                    market_id="MKT-01",
                    strategy_name=self.name,
                    signal_type=SignalType.HOLD,
                    outcome="yes",
                    confidence=0.5,
                )
            )
        return signals


# --- Fixtures ---


@pytest.fixture
def mock_market_data() -> MarketData:
    """Provides a default market data state."""
    return MarketData(
        market_id="MKT-01",
        order_book=OrderBook(
            bids=[PriceLevel(price=0.48, size=100)],
            asks=[PriceLevel(price=0.49, size=100)],  # Price is < 0.50
        ),
        recent_trades=[],
        details=MarketDetails(
            market_id="MKT-01",
            name="Test Market",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        ),
    )


@pytest.fixture
def strategy_engine() -> StrategyEngine:
    """Provides a StrategyEngine with the simple test strategy."""
    return StrategyEngine(strategies=[SimpleBuyStrategy()])


@pytest.fixture
def risk_manager() -> RiskManager:
    """Provides a full RiskManager setup."""
    portfolio = Portfolio(initial_balance=1000.0)
    sizer = FixedAmountSizer(default_amount_usdc=10.0)
    return RiskManager(portfolio=portfolio, sizer=sizer)


# --- Integration Test Case ---


def test_strategy_generates_buy_signal_and_risk_manager_creates_order(
    strategy_engine: StrategyEngine,
    risk_manager: RiskManager,
    mock_market_data: MarketData,
):
    # 1. ARRANGE: Create input data that will trigger a BUY
    # (The default fixture `mock_market_data` has ask price 0.49)
    ingestion_data = IngestionEngineOutput(
        timestamp=datetime.now(timezone.utc),
        market_data={"MKT-01": mock_market_data},
        external_data=[],
    )

    # 2. ACT (Module 2): Run the Strategy Engine
    signals = strategy_engine.process_data_tick(ingestion_data)

    # 3. ASSERT (Module 2 Output): Check that a BUY signal was created
    assert len(signals) == 1
    assert signals[0].signal_type == SignalType.BUY
    assert signals[0].market_id == "MKT-01"

    # 4. ACT (Module 3): Run the Risk Manager
    order_request = risk_manager.process_signal(signals[0], ingestion_data.market_data)

    # 5. ASSERT (Module 3 Output): Check that an OrderRequest was created
    assert order_request is not None
    assert order_request.market_id == "MKT-01"
    assert order_request.side == "buy"
    assert order_request.size == pytest.approx(10.0 / 0.49)  # 10 USD / 0.49 price


def test_strategy_generates_hold_signal_and_risk_manager_does_nothing(
    strategy_engine: StrategyEngine,
    risk_manager: RiskManager,
    mock_market_data: MarketData,
):
    # 1. ARRANGE: Create input data that will trigger a HOLD
    mock_market_data.order_book.asks = [PriceLevel(price=0.51, size=100)]
    ingestion_data = IngestionEngineOutput(
        timestamp=datetime.now(timezone.utc),
        market_data={"MKT-01": mock_market_data},
        external_data=[],
    )

    # 2. ACT (Module 2): Run the Strategy Engine
    signals = strategy_engine.process_data_tick(ingestion_data)

    # 3. ASSERT (Module 2 Output): Check that a HOLD signal was created
    assert len(signals) == 1
    assert signals[0].signal_type == SignalType.HOLD

    # 4. ACT (Module 3): Run the Risk Manager
    order_request = risk_manager.process_signal(signals[0], ingestion_data.market_data)

    # 5. ASSERT (Module 3 Output): Check that NO order was created
    assert order_request is None
