# tests/integration/test_request_flow.py

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Import your project's components
from trading_bot.core.database import Base
from trading_bot.core.enums import (
    OrderSide,
    OrderStatus,
    PositionStatus,
    SignalType,
)
from trading_bot.core.models import Position as PositionModel
from trading_bot.core.schemas import (
    ExecutionResult,
    MarketData,
    MarketDetails,
    OrderBook,
    PriceLevel,
    TradeSignal,
)
from trading_bot.risk_management.manager import RiskManager
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.risk_management.sizing.fixed_amount import FixedAmountSizer

# --- Fixtures for the Integration Test ---


@pytest.fixture(scope="module")
def db_session() -> Session:
    """
    Creates a new in-memory SQLite database session for the test module.
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)  # Create tables from your models
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def market_data_map() -> dict[str, MarketData]:
    """
    Provides a mock map of market data for testing.
    """
    return {
        "MKT-01": MarketData(
            market_id="MKT-01",
            order_book=OrderBook(
                bids=[PriceLevel(price=0.59, size=100)],
                asks=[PriceLevel(price=0.60, size=100)],
            ),
            recent_trades=[],
            details=MarketDetails(
                market_id="MKT-01",
                name="Test Market",
                end_date=datetime.now(timezone.utc),
                resolution_source="test",
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
    4. The Portfolio's in-memory and DB state are verified.
    5. A second (closing) trade is processed to check P&L.
    """

    # --- 1. ARRANGE (Initial Setup) ---
    portfolio = Portfolio(initial_balance_usdc=10000.0)

    sizer = FixedAmountSizer(default_amount_usdc=600.0)
    risk_manager = RiskManager(portfolio=portfolio, sizer=sizer)

    # --- 2. ACT (First Trade: BUY) ---

    # Create a BUY signal
    buy_signal = TradeSignal(
        market_id="MKT-01",
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

    assert pre_trade_state.available_balance_usdc == 9400.0  # 10000 - 600

    # --- 4. ACT (Simulate Execution Fill) ---

    # Create a fill result
    fill_result = ExecutionResult(
        order_id="order-001",
        status=OrderStatus.FILLED,
        filled_size=1000.0,
        avg_price=0.60,
        timestamp=datetime.now(timezone.utc),
    )

    #
    portfolio.update_order_status(db_session, fill_result)

    # --- 5. ASSERT (Post-Trade State & DB) ---

    # Check in-memory portfolio state
    post_buy_state = portfolio.get_state(market_data_map)
    assert portfolio._cash_balance == 9400.0
    assert len(post_buy_state.open_orders) == 0
    assert len(post_buy_state.positions) == 1

    position = post_buy_state.positions[0]
    assert position.market_id == "MKT-01"
    assert position.size == 1000.0
    assert position.entry_price == 0.60

    # P&L calculation (Unrealized P&L: 1000 * (0.59 - 0.60) = -10)
    assert post_buy_state.total_balance_usdc == 9990.0  # 9400 cash + (1000 * 0.59)
    assert post_buy_state.available_balance_usdc == 9400.0

    # Check database state
    db_pos = db_session.query(PositionModel).filter_by(market_id="MKT-01").one()
    assert db_pos.size == 1000.0
    assert db_pos.entry_price == 0.60
    assert db_pos.status == PositionStatus.OPEN

    # --- 6. ACT (Second Trade: SELL to close) ---

    # Update market data (price went up)
    market_data_map["MKT-01"].order_book.bids = [PriceLevel(price=0.70, size=100)]

    # Create a SELL signal (to partially close)
    sell_signal = TradeSignal(
        market_id="MKT-01",
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
        filled_size=order_request_2.size,
        avg_price=0.70,
        timestamp=datetime.now(timezone.utc),
    )
    portfolio.update_order_status(db_session, fill_result_2)

    # --- 7. ASSERT (Final State) ---

    # P&L = 857.14 * (0.70 - 0.60) = +85.71
    # Cash = 9400 + (857.14 * 0.70) + 85.71 = 9400 + 600 + 85.71 = 10085.71
    assert portfolio._cash_balance == pytest.approx(10000 + (60 / 0.7))

    final_state = portfolio.get_state(market_data_map)
    assert len(final_state.positions) == 1

    # Remaining position = 1000 - 857.14 = 142.86
    final_pos = final_state.positions[0]
    assert final_pos.size == pytest.approx(1000 - (600 / 0.7))
    assert final_pos.entry_price == 0.60

    # Check DB
    db_pos_final = db_session.query(PositionModel).filter_by(market_id="MKT-01").one()
    assert db_pos_final.size == pytest.approx(1000 - (600 / 0.7))
    assert db_pos_final.status == PositionStatus.OPEN
