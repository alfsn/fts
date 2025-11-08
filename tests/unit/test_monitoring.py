# tests/unit/test_monitoring.py

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

# Import the enums and schemas we need
from trading_bot.core.enums import AlertSeverity
from trading_bot.core.models import EventLog
from trading_bot.core.schemas import Alert

# Import the modules to test
from trading_bot.monitoring import logger as logger_module
from trading_bot.monitoring.abc import BaseAlerter
from trading_bot.monitoring.alerter import (
    AlerterService,
    DatabaseAlerter,
    LogAlerter,
)

# --- Fixtures ---


@pytest.fixture
def mock_db_session() -> MagicMock:
    """Mocks the SQLAlchemy Session."""
    db = MagicMock(spec=Session)
    return db


@pytest.fixture
def sample_alert() -> Alert:
    """Provides a sample alert object."""
    return Alert(
        message="Test alert message",
        severity=AlertSeverity.ERROR,
    )


# --- Tests for logger.py ---


# We patch 'settings' to provide a consistent LOG_LEVEL for the test
@patch("src.trading_bot.monitoring.logger.settings")
def test_setup_logging(mock_settings):
    """
    Tests that setup_logging correctly configures the root logger.
    """
    # --- Arrange ---
    # Configure the mock settings object
    mock_settings.LOG_LEVEL = "DEBUG"

    # Get the root logger
    root_logger = logging.getLogger()

    # Ensure a clean state by removing existing handlers
    root_logger.handlers.clear()

    # Store the original level to restore it later
    original_level = root_logger.level

    # --- Act ---
    logger_module.setup_logging()

    # --- Assert ---
    # 1. Check if the level was set correctly
    assert root_logger.level == logging.DEBUG

    # 2. Check if a handler was added
    assert len(root_logger.handlers) == 1
    handler = root_logger.handlers[0]

    # 3. Check if it's the correct type of handler (StreamHandler to stdout)
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream == sys.stdout

    # 4. Check if the handler's level is also set
    assert handler.level == logging.DEBUG

    # 5. Check if the formatter is set and has the correct format string
    assert handler.formatter is not None
    assert (
        handler.formatter._fmt
        == "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s"
    )

    # --- Cleanup ---
    root_logger.handlers.clear()
    root_logger.setLevel(original_level)


# --- Tests for alerter.py ---


class TestLogAlerter:
    """Tests the LogAlerter implementation."""

    def test_log_alerter_send_alert(self, sample_alert, caplog):
        """
        Tests that LogAlerter writes to the logging system at
        the correct level.
        """
        # --- Arrange ---
        alerter = LogAlerter()

        # Use caplog to capture log output
        with caplog.at_level(logging.ERROR):
            # --- Act ---
            success = alerter.send_alert(sample_alert)

            # --- Assert ---
            assert success is True
            # Check that the message was logged
            assert "Test alert message" in caplog.text
            # Check that it used the correct level (ERROR)
            assert caplog.records[0].levelname == "ERROR"


class TestDatabaseAlerter:
    """Tests the DatabaseAlerter implementation."""

    def test_db_alerter_send_alert_success(self, sample_alert, mock_db_session):
        """
        Tests that DatabaseAlerter successfully adds an EventLog
        model to the session and commits it.
        """
        # --- Arrange ---
        alerter = DatabaseAlerter()
        module_name = "TestModule"
        sample_alert.message = f"{module_name}: {sample_alert.message}"

        # --- Act ---
        success = alerter.send_alert(sample_alert, db=mock_db_session)

        # --- Assert ---
        assert success is True

        # 1. Check that db.add() was called once
        mock_db_session.add.assert_called_once()

        # 2. Get the object that was added
        added_object = mock_db_session.add.call_args[0][0]

        # 3. Verify the object's contents
        assert isinstance(added_object, EventLog)
        assert added_object.severity == AlertSeverity.ERROR
        assert added_object.message == sample_alert.message
        assert added_object.source_module == module_name  # Check simple parsing

        # 4. Check that the transaction was committed
        mock_db_session.commit.assert_called_once()
        mock_db_session.rollback.assert_not_called()

    def test_db_alerter_send_alert_failure(self, sample_alert, mock_db_session, caplog):
        """
        Tests that DatabaseAlerter handles a DB error, rolls back,
        and returns False.
        """
        # --- Arrange ---
        alerter = DatabaseAlerter()

        # Configure the mock session to raise an error on commit
        mock_db_session.commit.side_effect = Exception("Database connection lost")

        # --- Act ---
        with caplog.at_level(logging.ERROR):
            success = alerter.send_alert(sample_alert, db=mock_db_session)

        # --- Assert ---
        assert success is False

        # 1. Check that add and commit were called, but commit failed
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # 2. Check that the transaction was rolled back
        mock_db_session.rollback.assert_called_once()

        # 3. Check that an error was logged
        assert "Failed to write alert to database" in caplog.text
        assert "Database connection lost" in caplog.text


class TestAlerterService:
    """Tests the main AlerterService coordinator."""

    def test_service_notify(self, mock_db_session):
        """
        Tests that AlerterService.notify() correctly formats
        an Alert and calls all of its handlers.
        """
        # --- Arrange ---
        # Create mock handlers
        mock_handler_1 = MagicMock(spec=BaseAlerter)
        mock_handler_2 = MagicMock(spec=DatabaseAlerter)  # Mock as a DB alerter

        service = AlerterService(handlers=[mock_handler_1, mock_handler_2])

        message = "A critical failure occurred"
        severity = AlertSeverity.CRITICAL
        module = "ExecutionEngine"

        # --- Act ---
        service.notify(mock_db_session, message, severity, module)

        # --- Assert ---

        # 1. Check that handler 1 (LogAlerter) was called correctly
        mock_handler_1.send_alert.assert_called_once()
        alert_arg_1 = mock_handler_1.send_alert.call_args[0][0]
        assert isinstance(alert_arg_1, Alert)
        assert alert_arg_1.message == f"{module}: {message}"
        assert alert_arg_1.severity == severity

        # 2. Check that handler 2 (DatabaseAlerter) was called correctly
        # It should receive both the alert and the db session
        mock_handler_2.send_alert.assert_called_once()
        call_args_2 = mock_handler_2.send_alert.call_args
        alert_arg_2 = call_args_2[0][0]  # The alert object
        db_arg_2 = call_args_2[1]["db"]  # The db session

        assert isinstance(alert_arg_2, Alert)
        assert alert_arg_2.message == f"{module}: {message}"
        assert db_arg_2 == mock_db_session

    def test_service_handler_failure_isolation(self, mock_db_session, caplog):
        """
        Tests that if one handler fails, the service logs the
        error but continues to call other handlers.
        """
        # --- Arrange ---
        mock_handler_good = MagicMock(spec=BaseAlerter)
        mock_handler_bad = MagicMock(spec=BaseAlerter)

        # Configure the bad handler to raise an error
        mock_handler_bad.send_alert.side_effect = Exception("Telegram API down")

        service = AlerterService(handlers=[mock_handler_bad, mock_handler_good])

        # --- Act ---
        with caplog.at_level(logging.ERROR):
            service.notify(mock_db_session, "Test", AlertSeverity.INFO, "TestModule")

        # --- Assert ---
        # 1. Check that the bad handler was called (and failed)
        mock_handler_bad.send_alert.assert_called_once()

        # 2. Check that the error was logged
        assert "Alerter handler MagicMock failed" in caplog.text
        assert "Telegram API down" in caplog.text

        # 3. Check that the good handler was *still* called
        mock_handler_good.send_alert.assert_called_once()
