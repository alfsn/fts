from datetime import datetime, timezone

import pytest
from quant_core.enums import (
    AssetType,
    Geography,
    OrderSide,
    OrderStatus,
    PositionStatus,
    SignalType,
)
from quant_core.models import Instrument, TradFiDetails
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Import your project's components
from trading_bot.core.database import ResearchBase
from trading_bot.core.schemas import (
    ExecutionResult,
    Instrument,
    MarketData,
    OrderBook,
    PriceLevel,
    TradeSignal,
)
from trading_bot.risk_management.manager import RiskManager
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.risk_management.sizing.fixed_amount import FixedAmountSizer

# tests/integration/test_request_flow.py


# --- Fixtures for the Integration Test ---


@pytest.fixture(scope="module")
def db_session() -> Session:
    """
    Creates a new in-memory SQLite database session for the test module.
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    ResearchBase.metadata.create_all(engine)  # Create tables from your models
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        ResearchBase.metadata.drop_all(engine)


@pytest.fixture
def market_data_map() -> dict[str, MarketData]:
    """
    Provides a mock map of market data for testing.
    """
    return {
        "MKT-01": MarketData(
            instrument_id="MKT-01",
            order_book=OrderBook(
                bids=[PriceLevel(price=0.59, quantity=100)],
                asks=[PriceLevel(price=0.60, quantity=100)],
            ),
            recent_trades=[],
            details=Instrument(
                instrument_id="MKT-01",
                name="Test Market",
                details=TradFiDetails(
                    asset_type=AssetType.STOCK,
                    geography=Geography.US,
                    industry="Tech",
                    currency="USD",
                ),
            ),
        )
    }


# --- Integration Test Case ---


def test_full_trade_loop_signal_to_fill_to_state_update(
    db_session: Session, market_data_map: dict[str, MarketData]
):
    """
    Tests the full "happy path" integration:
    1. A Signal is processed by RiskManager.
    2. An OrderRequest is created.
    3. The Portfolio is updated with a (fake) ExecutionResult.
    4. The Portfolio's in-memory state is verified (positions are not stored: D19).
    5. A second (closing) trade is processed to check P&L.
    """

    portfolio = Portfolio(initial_balance=10000.0)

    sizer = FixedAmountSizer(default_amount_quote=600.0)
    risk_manager = RiskManager(portfolio=portfolio, sizer=sizer)

    # --- 2. ACT (First Trade: BUY) ---

    # Create a BUY signal
    buy_signal = TradeSignal(
        instrument_id="MKT-01",
        strategy_name="test_strat",
        signal_type=SignalType.BUY,
        confidence=0.7,  # Confidence for Kelly, ignored by FixedAmount
    )

    # Process the signal
    order_request = risk_manager.process_signal(buy_signal, market_data_map)
    assert order_request is not None

    portfolio.add_open_order("order-001", order_request)

    # --- 3. ASSERT (Pre-Trade State) ---

    pre_trade_state = portfolio.get_state(market_data_map)
    assert len(pre_trade_state.open_orders) == 1
    # Best ask is 0.60, sizer is 600 USDC. Size = 600 / 0.60 = 1000 shares

    assert pre_trade_state.cash_balances[0].available == 9400.0  # 10000 - 600

    # --- 4. ACT (Simulate Execution Fill) ---

    # Create a fill result
    fill_result = ExecutionResult(
        order_id="order-001",
        status=OrderStatus.FILLED,
        filled_quantity=1000.0,
        avg_price=0.60,
        timestamp=datetime.now(timezone.utc),
    )

    #
    portfolio.update_order_status(fill_result)

    # --- 5. ASSERT (Post-Trade State & DB) ---

    # Check in-memory portfolio state
    post_buy_state = portfolio.get_state(market_data_map)
    assert portfolio._cash_balance == 9400.0
    assert len(post_buy_state.open_orders) == 0
    assert len(post_buy_state.positions) == 1

    position = post_buy_state.positions[0]
    assert position.instrument_id == "MKT-01"
    assert position.quantity == 1000.0
    assert position.cost_basis == 0.60

    # Total is the marked equity (PR 7b): no bars yet, so the position is at
    # its entry price, 9400 + 1000 x 0.60. The bid (0.59) is for limits only.
    assert post_buy_state.cash_balances[0].total == 10000.0
    assert post_buy_state.cash_balances[0].available == 9400.0

    # --- 6. ACT (Second Trade: SELL to close) ---

    # Update market data (price went up)
    market_data_map["MKT-01"].order_book.bids = [PriceLevel(price=0.70, quantity=100)]

    # Create a SELL signal (to partially close)
    sell_signal = TradeSignal(
        instrument_id="MKT-01",
        strategy_name="test_strat",
        signal_type=SignalType.SELL,
        confidence=0.3,
    )

    # Sizer is 600 USDC. Best bid is 0.70. Size = 600 / 0.70 = 857.14 shares
    order_request_2 = risk_manager.process_signal(sell_signal, market_data_map)
    assert order_request_2 is not None
    assert order_request_2.side == OrderSide.SELL

    portfolio.add_open_order("order-002", order_request_2)

    # Simulate fill
    fill_result_2 = ExecutionResult(
        order_id="order-002",
        status=OrderStatus.FILLED,
        filled_quantity=order_request_2.quantity,
        avg_price=0.70,
        timestamp=datetime.now(timezone.utc),
    )
    portfolio.update_order_status(fill_result_2)

    # --- 7. ASSERT (Final State) ---

    # Cash = 9400 + (857.14 * 0.70) = 9400 + 600 = 10000.0
    assert portfolio._cash_balance == pytest.approx(10000.0)

    final_state = portfolio.get_state(market_data_map)
    assert len(final_state.positions) == 1

    # Remaining position = 1000 - 857.14 = 142.86
    final_pos = final_state.positions[0]
    assert final_pos.quantity == pytest.approx(1000 - (600 / 0.7))
    assert final_pos.cost_basis == 0.60
