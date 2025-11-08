# tests/unit/test_execution_engine.py

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# Import Schemas, Models, and Enums
from trading_bot.core.enums import MarketOutcome, OrderSide, OrderStatus
from trading_bot.core.models import OrderLog as OrderLogModel
from trading_bot.core.schemas import ExecutionResult, OrderRequest

# Import classes to be mocked
from trading_bot.execution.abc import BaseExecutionHandler

# Import the class we are testing
from trading_bot.execution.engine import ExecutionEngine
from trading_bot.risk_management.portfolio import Portfolio

# --- Pytest Fixtures ---


@pytest.fixture
def mock_handler() -> MagicMock:
    """Mocks the BaseExecutionHandler."""
    handler = MagicMock(spec=BaseExecutionHandler)
    handler.market_name.value = "mock_market"
    return handler


@pytest.fixture
def mock_portfolio() -> MagicMock:
    """Mocks the Portfolio."""
    return MagicMock(spec=Portfolio)


@pytest.fixture
def mock_db_session() -> MagicMock:
    """Mocks the SQLAlchemy Session and its nested transaction context."""
    db_session = MagicMock(spec=Session)

    # Mock the nested transaction context manager
    # `db.begin_nested()` returns a context manager.
    # We mock its `__enter__` and `__exit__` methods.
    mock_transaction = MagicMock()
    mock_transaction.__enter__.return_value = None

    # __exit__ must be a callable mock that returns None to propagate exceptions
    mock_transaction.__exit__ = MagicMock(return_value=None)

    db_session.begin_nested.return_value = mock_transaction

    return db_session


@pytest.fixture
def engine(
    mock_handler: BaseExecutionHandler, mock_portfolio: Portfolio
) -> ExecutionEngine:
    """Returns a standard ExecutionEngine instance with mocks."""
    return ExecutionEngine(
        execution_handler=mock_handler,
        portfolio=mock_portfolio,
        max_retry_attempts=3,
        enable_auto_reconciliation=True,
    )


@pytest.fixture
def sample_order_request() -> OrderRequest:
    """A sample order to be executed."""
    return OrderRequest(
        market_id="MKT_123",
        side=OrderSide.BUY,
        outcome=MarketOutcome.YES,
        size=100.0,
        price=0.50,
    )


@pytest.fixture
def sample_result_open() -> ExecutionResult:
    """A sample OPEN result from the handler."""
    return ExecutionResult(
        order_id="HANDLER_ORDER_001",
        status=OrderStatus.OPEN,
        filled_size=0,
        avg_price=0,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_result_filled() -> ExecutionResult:
    """A sample FILLED result from the handler."""
    return ExecutionResult(
        order_id="HANDLER_ORDER_002",
        status=OrderStatus.FILLED,
        filled_size=100.0,
        avg_price=0.50,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_order_log_open(sample_result_open: ExecutionResult) -> OrderLogModel:
    """A sample ORM model for an open order."""
    return OrderLogModel(
        order_id=sample_result_open.order_id,
        status=OrderStatus.OPEN,
        market_id="MKT_123",
        strategy_name="test_strat",
        side=OrderSide.BUY,
        outcome=MarketOutcome.YES,
        requested_size=100.0,
        requested_price=0.50,
        filled_size=0,
        avg_fill_price=0,
    )


# --- Test Cases ---


class TestExecutionEngine:
    """Unit tests for the ExecutionEngine."""

    def test_initialization(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_portfolio: MagicMock,
    ):
        """Tests that the engine is initialized correctly."""
        assert engine.handler == mock_handler
        assert engine.portfolio == mock_portfolio
        assert engine.max_retry_attempts == 3
        assert engine.enable_auto_reconciliation is True
        assert engine._reconciliation_queue == []

    # --- execute_order Tests ---

    def test_execute_order_happy_path_open(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_request: OrderRequest,
        sample_result_open: ExecutionResult,
    ):
        """
        Tests the successful execution of an order that becomes OPEN.
        1. Handler returns OPEN result.
        2. Database logs the new order.
        3. Portfolio is updated to track the open order.
        """
        # --- Arrange ---
        mock_handler.execute_order.return_value = sample_result_open
        # Mock DB to find no existing order
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        # --- Act ---
        result = engine.execute_order(
            sample_order_request, mock_db_session, "test_strat"
        )

        # --- Assert ---
        # 1. Handler was called
        mock_handler.execute_order.assert_called_once_with(sample_order_request)

        # 2. Database was updated
        mock_db_session.begin_nested.assert_called_once()
        mock_db_session.add.assert_called_once()  # For the new OrderLogModel
        mock_db_session.commit.assert_called_once()

        # 3. Portfolio was updated
        mock_portfolio.add_open_order.assert_called_once_with(
            sample_result_open.order_id, sample_order_request
        )
        mock_portfolio.update_order_status.assert_not_called()

        # 4. Result is correct
        assert result == sample_result_open

    def test_execute_order_happy_path_filled(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_request: OrderRequest,
        sample_result_filled: ExecutionResult,
    ):
        """
        Tests the successful execution of an order that is FILLED immediately.
        1. Handler returns FILLED result.
        2. Database logs the new, filled order.
        3. Portfolio is updated with the fill.
        """
        # --- Arrange ---
        mock_handler.execute_order.return_value = sample_result_filled
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        # --- Act ---
        result = engine.execute_order(
            sample_order_request, mock_db_session, "test_strat"
        )

        # --- Assert ---
        # 1. Handler was called
        mock_handler.execute_order.assert_called_once_with(sample_order_request)

        # 2. Database was updated
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # 3. Portfolio was updated
        mock_portfolio.add_open_order.assert_not_called()
        mock_portfolio.update_order_status.assert_called_once_with(
            mock_db_session, sample_result_filled
        )

        # 4. Result is correct
        assert result == sample_result_filled

    def test_execute_order_handler_failure(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_request: OrderRequest,
        caplog: pytest.LogCaptureFixture,
    ):
        """
        Tests the case where the handler.execute_order() call fails.
        1. Handler raises an exception.
        2. A synthetic FAILED result is created.
        3. The FAILED result is logged to the DB.
        4. The FAILED result is passed to the portfolio.
        """
        # --- Arrange ---
        mock_handler.execute_order.side_effect = ValueError("API is down")

        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        # --- Act ---
        with caplog.at_level(logging.ERROR):
            result = engine.execute_order(
                sample_order_request, mock_db_session, "test_strat"
            )

        # --- Assert ---
        # 1. Log message was generated
        assert "Execution API call failed" in caplog.text
        assert "API is down" in caplog.text

        # 2. Synthetic result was created
        assert result.status == OrderStatus.FAILED
        assert result.order_id.startswith("local_fail_handler_failure_")

        # 3. Database was updated with the FAILED result
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # 4. Portfolio was updated with the FAILED result
        mock_portfolio.add_open_order.assert_not_called()
        mock_portfolio.update_order_status.assert_called_once_with(
            mock_db_session, result
        )

    def test_execute_order_database_failure(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_request: OrderRequest,
        sample_result_open: ExecutionResult,
        caplog: pytest.LogCaptureFixture,
    ):
        """
        Tests the critical failure where the order is PLACED, but the DB log fails.
        1. Handler returns OPEN result.
        2. Database commit raises SQLAlchemyError.
        3. The error is logged as CRITICAL.
        4. The order is queued for reconciliation.
        5. The portfolio is NOT updated (to prevent desync).
        """
        # --- Arrange ---
        mock_handler.execute_order.return_value = sample_result_open

        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        # Simulate DB failure
        mock_db_session.commit.side_effect = SQLAlchemyError("DB Connection Lost")

        # --- Act ---
        with caplog.at_level(logging.CRITICAL):
            result = engine.execute_order(
                sample_order_request, mock_db_session, "test_strat"
            )

        # --- Assert ---
        # 1. Critical log was generated
        assert "DATABASE ERROR: Failed to log order" in caplog.text
        assert "DB Connection Lost" in caplog.text

        # 2. DB transaction was rolled back
        mock_db_session.rollback.assert_called_once()

        # 3. Portfolio was NOT updated
        mock_portfolio.add_open_order.assert_not_called()
        mock_portfolio.update_order_status.assert_not_called()

        # 4. Reconciliation was queued
        assert len(engine._reconciliation_queue) == 1
        assert (
            engine._reconciliation_queue[0]["order_id"] == sample_result_open.order_id
        )

        # 5. Original result is returned
        assert result == sample_result_open

    def test_execute_order_portfolio_failure(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_request: OrderRequest,
        sample_result_filled: ExecutionResult,
        caplog: pytest.LogCaptureFixture,
    ):
        """
        Tests failure where DB log succeeds, but in-memory Portfolio update fails.
        1. Handler returns FILLED result.
        2. Database log succeeds.
        3. Portfolio update raises an Exception.
        4. The error is logged.
        5. The order is queued for reconciliation.
        """
        # --- Arrange ---
        mock_handler.execute_order.return_value = sample_result_filled
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        # Simulate portfolio failure
        mock_portfolio.update_order_status.side_effect = ValueError("Portfolio Bug")

        # --- Act ---
        with caplog.at_level(logging.ERROR):
            result = engine.execute_order(
                sample_order_request, mock_db_session, "test_strat"
            )

        # --- Assert ---
        # 1. Database update succeeded
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # 2. Portfolio update was called
        mock_portfolio.update_order_status.assert_called_once_with(
            mock_db_session, sample_result_filled
        )

        # 3. Error was logged
        assert "Portfolio update failed" in caplog.text
        assert "Portfolio Bug" in caplog.text

        # 4. Reconciliation was queued
        assert len(engine._reconciliation_queue) == 1
        assert (
            engine._reconciliation_queue[0]["order_id"] == sample_result_filled.order_id
        )

        # 5. Original result is returned
        assert result == sample_result_filled

    # --- check_order_status Tests ---

    def test_check_order_status_no_change(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_log_open: OrderLogModel,
    ):
        """Tests checking an order whose status has not changed."""
        # --- Arrange ---
        order_id = sample_order_log_open.order_id
        # Handler returns the same OPEN status
        mock_handler.get_order_status.return_value = sample_order_log_open
        # DB finds the existing OPEN log
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            sample_order_log_open
        )

        # --- Act ---
        result = engine.check_order_status(order_id, mock_db_session)

        # --- Assert ---
        mock_handler.get_order_status.assert_called_once_with(order_id)
        # No updates should happen
        mock_db_session.commit.assert_not_called()
        mock_portfolio.update_order_status.assert_not_called()
        assert result.status == OrderStatus.OPEN

    def test_check_order_status_changed_to_filled(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_log_open: OrderLogModel,
        sample_result_filled: ExecutionResult,
    ):
        """Tests an order that transitions from OPEN to FILLED."""
        # --- Arrange ---
        order_id = sample_order_log_open.order_id
        # Use the same order_id for the filled result
        filled_result = sample_result_filled.model_copy(update={"order_id": order_id})

        mock_handler.get_order_status.return_value = filled_result
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            sample_order_log_open
        )

        # --- Act ---
        result = engine.check_order_status(order_id, mock_db_session)

        # --- Assert ---
        # 1. Handler was called
        mock_handler.get_order_status.assert_called_once_with(order_id)

        # 2. Database log was updated
        mock_db_session.commit.assert_called_once()
        assert sample_order_log_open.status == OrderStatus.FILLED
        assert sample_order_log_open.filled_size == filled_result.filled_size

        # 3. Portfolio was updated
        mock_portfolio.update_order_status.assert_called_once_with(
            mock_db_session, filled_result
        )

        # 4. Result is correct
        assert result == filled_result

    def test_check_order_status_reconcile_missing_log(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        sample_result_filled: ExecutionResult,
        caplog: pytest.LogCaptureFixture,
    ):
        """
        Tests the reconciliation logic when an order is found on the
        exchange but is missing from the local database.
        """
        # --- Arrange ---
        order_id = sample_result_filled.order_id
        mock_handler.get_order_status.return_value = sample_result_filled
        # DB finds no log
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        # --- Act ---
        with caplog.at_level(logging.WARNING):
            result = engine.check_order_status(order_id, mock_db_session)

        # --- Assert ---
        # 1. Log message was generated
        assert "exists on exchange but not in database" in caplog.text

        # 2. Reconciliation was attempted (best-effort)
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # 3. The new log should have reconciled fields
        added_log: OrderLogModel = mock_db_session.add.call_args[0][0]
        assert added_log.order_id == order_id
        assert added_log.strategy_name == "reconciled"
        assert added_log.status == OrderStatus.FILLED

        # 4. Result is returned
        assert result == sample_result_filled

    # --- cancel_order Tests ---

    def test_cancel_order_success(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_log_open: OrderLogModel,
    ):
        """Tests a successful order cancellation."""
        # --- Arrange ---
        order_id = sample_order_log_open.order_id
        cancelled_result = ExecutionResult(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            filled_size=0,
            avg_price=0,
            timestamp=datetime.now(timezone.utc),
        )

        mock_handler.cancel_order.return_value = cancelled_result
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            sample_order_log_open
        )

        # --- Act ---
        result = engine.cancel_order(order_id, mock_db_session)

        # --- Assert ---
        # 1. Handler was called
        mock_handler.cancel_order.assert_called_once_with(order_id)

        # 2. DB log was updated
        mock_db_session.commit.assert_called_once()
        assert sample_order_log_open.status == OrderStatus.CANCELLED

        # 3. Portfolio was updated
        mock_portfolio.update_order_status.assert_called_once_with(
            mock_db_session, cancelled_result
        )

        # 4. Result is correct
        assert result == cancelled_result

    def test_cancel_order_handler_failure(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_log_open: OrderLogModel,
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests when the handler.cancel_order() call fails."""
        # --- Arrange ---
        order_id = sample_order_log_open.order_id
        mock_handler.cancel_order.side_effect = ValueError("Cancel API Failed")
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            sample_order_log_open
        )

        # --- Act ---
        with caplog.at_level(logging.ERROR):
            result = engine.cancel_order(order_id, mock_db_session)

        # --- Assert ---
        # 1. Error was logged
        assert "Cancel API call" in caplog.text
        assert "Cancel API Failed" in caplog.text

        # 2. Synthetic FAILED result was created
        assert result.status == OrderStatus.FAILED
        assert result.order_id == order_id

        # 3. DB log was updated to FAILED
        mock_db_session.commit.assert_called_once()
        assert sample_order_log_open.status == OrderStatus.FAILED

        # 4. Portfolio was updated with FAILED status
        mock_portfolio.update_order_status.assert_called_once_with(
            mock_db_session, result
        )

    # --- Reconciliation Queue Test ---

    def test_reconciliation_queue_and_process(
        self,
        engine: ExecutionEngine,
        mock_handler: MagicMock,
        mock_db_session: MagicMock,
        mock_portfolio: MagicMock,
        sample_order_request: OrderRequest,
        sample_result_open: ExecutionResult,
        sample_result_filled: ExecutionResult,
    ):
        """
        Tests the full reconciliation loop:
        1. An order fails to log to DB and gets queued.
        2. reconcile_all_orders is called.
        3. The order is successfully re-polled, logged, and updated.
        """
        # --- 1. Queue the order (Simulate DB failure) ---
        mock_handler.execute_order.return_value = sample_result_open

        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        mock_db_session.commit.side_effect = SQLAlchemyError("DB Down")

        engine.execute_order(sample_order_request, mock_db_session, "test_strat")

        # Assert it was queued
        assert len(engine._reconciliation_queue) == 1
        queued_item = engine._reconciliation_queue[0]
        assert queued_item["order_id"] == sample_result_open.order_id

        # Reset mocks for next phase
        mock_db_session.commit.side_effect = None
        mock_db_session.reset_mock()
        mock_portfolio.reset_mock()

        # --- 2. Run reconciliation ---

        # Arrange mocks for reconcile_all_orders
        order_id = sample_result_open.order_id
        # Handler now says the order is FILLED
        filled_result = sample_result_filled.model_copy(update={"order_id": order_id})
        mock_handler.get_order_status.return_value = filled_result

        # DB finds the log (we'll pretend it got created but update failed)
        # A more realistic test: DB *doesn't* find it, and we create it.
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            None  # Simulate it was never created
        )

        # --- Act ---
        recon_results = engine.reconcile_all_orders(mock_db_session)

        # --- Assert ---
        # 1. Handler was polled
        mock_handler.get_order_status.assert_called_once_with(order_id)

        # 2. DB was updated (reconcile_missing_order was called)
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # 3. Portfolio was updated
        mock_portfolio.update_order_status.assert_called_once_with(
            mock_db_session, filled_result
        )

        # 4. Queue is now empty
        assert len(engine._reconciliation_queue) == 0
        assert recon_results["success"] == 1
        assert recon_results["failed"] == 0
