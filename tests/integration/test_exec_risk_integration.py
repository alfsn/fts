# tests/integration/test_exec_risk_integration.py

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trading_bot.core.database import Base
from trading_bot.core.enums import (
    OrderSide,
    OrderStatus,
    SignalType,
)
from trading_bot.core.models import OrderLog as OrderLogModel
from trading_bot.core.models import Position as PositionModel
from trading_bot.core.schemas import (
    ExecutionResult,
    MarketData,
    MarketDetails,
    OrderBook,
    PriceLevel,
    TradeSignal,
)
from trading_bot.execution.abc import BaseExecutionHandler
from trading_bot.execution.engine import ExecutionEngine
from trading_bot.risk_management.manager import RiskManager
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.risk_management.sizing.fixed_amount import FixedAmountSizer

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Fixtures for Real Components ---


@pytest.fixture(scope="function")
def in_memory_db_session() -> Session:
    """
    Creates a real in-memory SQLite database session for the test.
    This ensures that database transactions, object creation, and
    rollbacks are tested properly.
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)  # Create all tables (OrderLog, Position, etc.)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def mock_handler() -> MagicMock:
    """Mocks the BaseExecutionHandler."""
    handler = MagicMock(spec=BaseExecutionHandler)
    handler.market_name.value = "mock_market"
    return handler


@pytest.fixture
def real_portfolio() -> Portfolio:
    """Returns a real Portfolio instance with 10,000 USDC."""
    return Portfolio(initial_balance=10000.0)


@pytest.fixture
def real_sizer() -> FixedAmountSizer:
    """Returns a real FixedAmountSizer of 100 quote currency units."""
    return FixedAmountSizer(default_amount_quote=100.0)


@pytest.fixture
def real_risk_manager(
    real_portfolio: Portfolio, real_sizer: FixedAmountSizer
) -> RiskManager:
    """Returns a real RiskManager wired to the real portfolio and sizer."""
    return RiskManager(
        portfolio=real_portfolio,
        sizer=real_sizer,
        max_allocation_per_market=0.5,  # 50%
        max_total_positions=10,
    )


@pytest.fixture
def real_execution_engine(
    mock_handler: MagicMock, real_portfolio: Portfolio
) -> ExecutionEngine:
    """Returns a real ExecutionEngine wired to the real portfolio and mock handler."""
    return ExecutionEngine(
        execution_handler=mock_handler,
        portfolio=real_portfolio,
    )


# --- Fixtures for Sample Data ---


@pytest.fixture
def mock_market_data() -> MarketData:
    """Returns a mock MarketData object with a live order book."""
    return MarketData(
        market_id="MARKET_01",
        order_book=OrderBook(
            bids=[PriceLevel(price=0.49, size=100)],
            asks=[PriceLevel(price=0.51, size=100)],  # We will BUY at 0.51
        ),
        recent_trades=[],
        details=MarketDetails(
            market_id="MARKET_01",
            name="Test Market",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        ),
    )


@pytest.fixture
def market_data_map(mock_market_data: MarketData) -> dict:
    """Returns a map containing the mock market data."""
    return {"MARKET_01": mock_market_data}


@pytest.fixture
def mock_buy_signal() -> TradeSignal:
    """Returns a simple BUY signal for MARKET_01."""
    return TradeSignal(
        market_id="MARKET_01",
        strategy_name="integration_test_strat",
        signal_type=SignalType.BUY,
        confidence=0.7,  # High confidence
    )


# --- Integration Test Case ---


def test_full_order_lifecycle_integration(
    real_risk_manager: RiskManager,
    real_execution_engine: ExecutionEngine,
    real_portfolio: Portfolio,
    mock_handler: MagicMock,
    in_memory_db_session: Session,
    mock_buy_signal: TradeSignal,
    market_data_map: dict,
):
    """
    Tests the full data flow from a Signal to a Filled Position.

    1.  [Flow 1] RiskManager processes a Signal -> creates OrderRequest.
    2.  [Flow 1] ExecutionEngine takes OrderRequest -> "places" it (mocked)
                 -> logs 'OPEN' to DB -> updates Portfolio state.
    3.  [Flow 2] ExecutionEngine checks for updates -> "finds" a Fill (mocked).
    4.  [Flow 2] ExecutionEngine logs 'FILLED' to DB -> updates Portfolio
                 with the new Position and cash balance.
    """
    db = in_memory_db_session
    strategy_name = mock_buy_signal.strategy_name
    order_id = "INTEGRATION_ORDER_001"

    # --- 1. ARRANGE MOCKS ---

    # Mock the two-step lifecycle: OPEN -> FILLED
    open_result = ExecutionResult(
        order_id=order_id,
        status=OrderStatus.OPEN,
        filled_size=0,
        avg_price=0,
        timestamp=datetime.now(timezone.utc),
    )
    filled_result = ExecutionResult(
        order_id=order_id,
        status=OrderStatus.FILLED,
        filled_size=196.0784,  # 100 / 0.51
        avg_price=0.51,
        timestamp=datetime.now(timezone.utc),
    )

    # `execute_order` returns the OPEN result
    mock_handler.execute_order.return_value = open_result

    # `get_order_status` will return OPEN first, then FILLED on the second call
    mock_handler.get_order_status.side_effect = [
        open_result,
        filled_result,
    ]

    # --- 2. ACT (Flow 1: RiskManager -> ExecutionEngine) ---
    logger.info("--- Flow 1: Processing Signal to Open Order ---")

    # RiskManager processes the signal and creates the order
    order_request = real_risk_manager.process_signal(mock_buy_signal, market_data_map)

    # Asserts for the OrderRequest
    assert order_request is not None
    assert order_request.side == OrderSide.BUY
    assert order_request.price == 0.51
    assert order_request.size == pytest.approx(100.0 / 0.51)

    # ExecutionEngine executes the order
    exec_result_open = real_execution_engine.execute_order(
        order_request, db, strategy_name
    )

    # --- 3. ASSERT (Flow 1: State is OPEN) ---
    logger.info("--- Asserting Flow 1: Order is OPEN ---")

    # Check that the handler was called correctly
    mock_handler.execute_order.assert_called_once_with(order_request)
    assert exec_result_open == open_result

    # Check the database: The order should be logged as OPEN
    db_log = db.query(OrderLogModel).filter_by(order_id=order_id).first()
    assert db_log is not None
    assert db_log.status == OrderStatus.OPEN
    assert db_log.strategy_name == strategy_name
    assert db_log.requested_price == 0.51

    # Check portfolio state: Cash should be committed, order tracked as open
    portfolio_state_open = real_portfolio.get_state(market_data_map)
    assert len(portfolio_state_open.open_orders) == 1
    assert portfolio_state_open.open_orders[0].market_id == "MARKET_01"
    assert portfolio_state_open.total_balance_quote == 10000.0  # No P&L yet
    # Available balance = 10000 - (100 / 0.51) * 0.51 = 9900.0
    assert portfolio_state_open.available_balance_quote == pytest.approx(9900.0)
    assert len(portfolio_state_open.positions) == 0  # No position yet

    # --- 4. ACT (Flow 2: ExecutionEngine -> Portfolio) ---
    logger.info("--- Flow 2: Polling for Fill -> Updating Portfolio ---")

    # First poll: no change
    exec_result_poll_1 = real_execution_engine.check_order_status(order_id, db)
    assert exec_result_poll_1.status == OrderStatus.OPEN

    # Second poll: order is now FILLED
    exec_result_poll_2 = real_execution_engine.check_order_status(order_id, db)
    assert exec_result_poll_2.status == OrderStatus.FILLED

    # --- 5. ASSERT (Flow 2: State is FILLED) ---
    logger.info("--- Asserting Flow 2: Order is FILLED ---")

    # Check that the handler was called
    assert mock_handler.get_order_status.call_count == 2

    # Check the database: The log should be updated to FILLED
    db.refresh(db_log)  # Refresh object from DB
    assert db_log.status == OrderStatus.FILLED
    assert db_log.filled_size == filled_result.filled_size

    # Check database: A new PositionModel should be created
    db_pos = (
        db.query(PositionModel)
        .filter_by(
            market_id="MARKET_01",
        )
        .first()
    )
    assert db_pos is not None
    assert db_pos.size == filled_result.filled_size
    assert db_pos.entry_price == 0.51

    # Check final portfolio state: Order is closed, position is open
    portfolio_state_filled = real_portfolio.get_state(market_data_map)
    assert len(portfolio_state_filled.open_orders) == 0
    assert len(portfolio_state_filled.positions) == 1
    assert portfolio_state_filled.positions[0].size == filled_result.filled_size
    assert portfolio_state_filled.positions[0].entry_price == 0.51

    # Check cash: 10000 (start) - 100 (fill cost) = 9900
    assert real_portfolio._cash_balance == pytest.approx(9900.0)
    # Available and total should be equal now (minus unrealized P&L)
    assert portfolio_state_filled.available_balance_quote == pytest.approx(9900.0)

    # Total balance = 9900 (cash) + (position_size * current_bid_price)
    # position_size = 196.0784
    # current_bid_price = 0.49 (from market_data_map)
    # market_value = 196.0784 * 0.49 = 96.0784
    # total_balance = 9900 + 96.0784 = 9996.0784
    # This reflects the immediate unrealized loss from crossing the spread.
    assert portfolio_state_filled.total_balance_quote == pytest.approx(
        9900.0 + (filled_result.filled_size * 0.49)
    )
    logger.info("--- Integration test successful ---")


def test_order_rejection_integration(
    real_risk_manager: RiskManager,
    real_execution_engine: ExecutionEngine,
    real_portfolio: Portfolio,
    mock_handler: MagicMock,
    in_memory_db_session: Session,
    mock_buy_signal: TradeSignal,
    market_data_map: dict,
):
    """
    Tests the flow where an order is immediately REJECTED by the handler.

    1.  [Flow 1] RiskManager processes a Signal -> creates OrderRequest.
    2.  [Flow 1] ExecutionEngine takes OrderRequest -> "places" it (mocked).
    3.  [Flow 1] MockHandler returns ExecutionResult(status=REJECTED).
    4.  [Flow 1] ExecutionEngine logs 'REJECTED' to DB.
    5.  [Flow 1] ExecutionEngine updates Portfolio, which should result in no
                 open orders and no change in cash.
    """
    db = in_memory_db_session
    strategy_name = mock_buy_signal.strategy_name
    order_id = "REJECTED_ORDER_001"

    # --- 1. ARRANGE MOCKS ---
    rejected_result = ExecutionResult(
        order_id=order_id,
        status=OrderStatus.REJECTED,
        filled_size=0,
        avg_price=0,
        timestamp=datetime.now(timezone.utc),
    )
    mock_handler.execute_order.return_value = rejected_result

    # --- 2. ACT (RiskManager -> ExecutionEngine) ---
    logger.info("--- Flow: Processing Signal to Rejected Order ---")
    order_request = real_risk_manager.process_signal(mock_buy_signal, market_data_map)
    assert order_request is not None

    exec_result_rejected = real_execution_engine.execute_order(
        order_request, db, strategy_name
    )

    # --- 3. ASSERT (State is REJECTED) ---
    logger.info("--- Asserting Flow: Order is REJECTED ---")

    # Check that the handler was called and returned the rejection
    mock_handler.execute_order.assert_called_once_with(order_request)
    assert exec_result_rejected == rejected_result

    # Check the database: The order should be logged as REJECTED
    db_log = db.query(OrderLogModel).filter_by(order_id=order_id).first()
    assert db_log is not None
    assert db_log.status == OrderStatus.REJECTED

    # Check portfolio state: No changes
    portfolio_state_rejected = real_portfolio.get_state(market_data_map)
    assert len(portfolio_state_rejected.open_orders) == 0
    assert len(portfolio_state_rejected.positions) == 0
    assert portfolio_state_rejected.available_balance_quote == 10000.0
    assert portfolio_state_rejected.total_balance_quote == 10000.0
    assert real_portfolio._cash_balance == 10000.0


def test_order_cancellation_integration(
    real_risk_manager: RiskManager,
    real_execution_engine: ExecutionEngine,
    real_portfolio: Portfolio,
    mock_handler: MagicMock,
    in_memory_db_session: Session,
    mock_buy_signal: TradeSignal,
    market_data_map: dict,
):
    """
    Tests the flow where an OPEN order is successfully CANCELLED.

    1.  [Setup] An order is successfully placed and is OPEN.
    2.  [Flow 1] ExecutionEngine.cancel_order() is called.
    3.  [Flow 1] MockHandler returns ExecutionResult(status=CANCELLED).
    4.  [Flow 1] ExecutionEngine logs 'CANCELLED' to DB.
    5.  [Flow 1] ExecutionEngine updates Portfolio, which should "un-commit"
                 the cash and remove the open order.
    """
    db = in_memory_db_session
    strategy_name = mock_buy_signal.strategy_name
    order_id = "CANCEL_ORDER_001"

    # --- 1. ARRANGE MOCKS ---
    open_result = ExecutionResult(
        order_id=order_id,
        status=OrderStatus.OPEN,
        filled_size=0,
        avg_price=0,
        timestamp=datetime.now(timezone.utc),
    )
    cancelled_result = ExecutionResult(
        order_id=order_id,
        status=OrderStatus.CANCELLED,
        filled_size=0,
        avg_price=0,
        timestamp=datetime.now(timezone.utc),
    )
    mock_handler.execute_order.return_value = open_result
    mock_handler.cancel_order.return_value = cancelled_result

    # --- 2. ACT (Step 1: Open the order) ---
    logger.info("--- Setup: Processing Signal to Open Order ---")
    order_request = real_risk_manager.process_signal(mock_buy_signal, market_data_map)
    assert order_request is not None
    real_execution_engine.execute_order(order_request, db, strategy_name)

    # --- 3. ASSERT (Step 1: Check if OPEN) ---
    logger.info("--- Asserting Setup: Order is OPEN ---")
    portfolio_state_open = real_portfolio.get_state(market_data_map)
    assert len(portfolio_state_open.open_orders) == 1
    assert portfolio_state_open.available_balance_quote == pytest.approx(9900.0)
    db_log = db.query(OrderLogModel).filter_by(order_id=order_id).first()
    assert db_log is not None
    assert db_log.status == OrderStatus.OPEN

    # --- 4. ACT (Step 2: Cancel the order) ---
    logger.info("--- Flow: Cancelling the Open Order ---")
    exec_result_cancelled = real_execution_engine.cancel_order(order_id, db)

    # --- 5. ASSERT (Step 2: Check if CANCELLED) ---
    logger.info("--- Asserting Flow: Order is CANCELLED ---")

    # Check that the handler was called and returned the cancellation
    mock_handler.cancel_order.assert_called_once_with(order_id)
    assert exec_result_cancelled == cancelled_result

    # Check the database: The log should be updated to CANCELLED
    db.refresh(db_log)
    assert db_log.status == OrderStatus.CANCELLED

    # Check portfolio state: Cash is released, no open orders
    portfolio_state_cancelled = real_portfolio.get_state(market_data_map)
    assert len(portfolio_state_cancelled.open_orders) == 0
    assert len(portfolio_state_cancelled.positions) == 0
    assert portfolio_state_cancelled.available_balance_quote == 10000.0
    assert portfolio_state_cancelled.total_balance_quote == 10000.0
    assert real_portfolio._cash_balance == 10000.0
