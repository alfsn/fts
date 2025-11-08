# src/trading_bot/monitoring/alerter.py

"""
Provides the AlerterService and concrete implementations of BaseAlerter.

This module allows the bot to send notifications to multiple destinations
(like the console and the database) simultaneously.
"""

import logging
from typing import List

from sqlalchemy.orm import Session

from ..core.enums import AlertSeverity
from ..core.models import EventLog
from ..core.schemas import Alert
from .abc import BaseAlerter

logger = logging.getLogger(__name__)


class LogAlerter(BaseAlerter):
    """
    A simple alerter implementation that writes alerts to the
    standard logging system.
    """

    def send_alert(self, alert: Alert) -> bool:
        """
        Logs the alert message using the appropriate log level.

        :param alert: The Alert object to log.
        :return: True
        """
        log_level = getattr(logging, alert.severity.upper(), logging.INFO)
        logger.log(log_level, f"[ALERT] {alert.message}")
        return True


class DatabaseAlerter(BaseAlerter):
    """
    An alerter implementation that persists alerts to the
    `event_logs` table in the database.
    """

    def send_alert(self, alert: Alert, db: Session) -> bool:
        """
        Writes the alert to the EventLog table.

        :param alert: The Alert object to persist.
        :param db: The SQLAlchemy database session.
        :return: True if successful, False otherwise.
        """
        try:
            # Create the SQLAlchemy ORM model from the Pydantic schema
            log_entry = EventLog(
                severity=alert.severity,
                message=alert.message,
                source_module=alert.message.split(":")[0],  # Simple module parsing
            )
            db.add(log_entry)
            db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to write alert to database: {e}", exc_info=True)
            db.rollback()
            return False


class AlerterService:
    """
    Coordinates all active alerter implementations.

    This service is intended to be injected (using Dependency Injection)
    into other core modules like the ExecutionEngine or RiskManager,
    providing a single `notify` method.
    """

    def __init__(self, handlers: List[BaseAlerter]):
        """
        Initializes the service with a list of concrete alerter handlers.

        :param handlers: A list of objects that implement BaseAlerter
                         (e.g., [LogAlerter(), DatabaseAlerter()]).
        """
        self.handlers = handlers
        logger.info(f"AlerterService initialized with {len(handlers)} handlers.")

    def notify(
        self,
        db: Session,
        message: str,
        severity: AlertSeverity,
        module: str = "System",
    ):
        """
        Creates an Alert and sends it to all configured handlers.

        :param db: The SQLAlchemy database session (required by DatabaseAlerter).
        :param message: The text of the alert message.
        :param severity: The severity (e.g., INFO, ERROR, CRITICAL).
        :param module: The name of the module originating the alert.
        """
        full_message = f"{module}: {message}"
        alert = Alert(message=full_message, severity=severity)

        for handler in self.handlers:
            try:
                if isinstance(handler, DatabaseAlerter):
                    handler.send_alert(alert, db=db)
                else:
                    handler.send_alert(alert)
            except Exception as e:
                # Log failure of one handler but continue with others
                logger.error(
                    f"Alerter handler {type(handler).__name__} failed: {e}",
                    exc_info=True,
                )
