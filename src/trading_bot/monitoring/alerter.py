# src/trading_bot/monitoring/alerter.py

"""
Provides the AlerterService and concrete implementations of BaseAlerter.

This module allows the bot to send notifications to multiple destinations
(like the console and the database) simultaneously.
"""

import logging
from typing import List

from sqlalchemy.orm import Session, sessionmaker

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

    def __init__(self, session_factory: sessionmaker[Session]):
        """
        Initializes the alerter with a session factory.

        :param session_factory: The SQLAlchemy SessionLocal factory.
        """
        self.session_factory = session_factory

    def send_alert(self, alert: Alert) -> bool:
        """
        Writes the alert to the EventLog table.
        (Note: The `db` parameter is gone, matching the BaseAlerter)

        :param alert: The Alert object to persist.
        :return: True if successful, False otherwise.
        """
        # Create a new, short-lived session just for this task
        db = self.session_factory()
        try:
            log_entry = EventLog(
                severity=alert.severity,
                message=alert.message,
                # Simple parsing. You could pass 'module' in the Alert schema too.
                source_module=alert.message.split(":")[0],
            )
            db.add(log_entry)
            db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to write alert to database: {e}", exc_info=True)
            db.rollback()
            return False
        finally:
            db.close()


class AlerterService:
    """
    Coordinates all active alerter implementations.
    ...
    """

    def __init__(self, handlers: List[BaseAlerter]):
        self.handlers = handlers
        logger.info(f"AlerterService initialized with {len(handlers)} handlers.")

    def notify(
        self,
        # The `db` parameter is removed. It's no longer this service's job.
        message: str,
        severity: AlertSeverity,
        module: str = "System",
    ):
        """
        Creates an Alert and sends it to all configured handlers.
        """
        full_message = f"{module}: {message}"
        alert = Alert(message=full_message, severity=severity)

        # This loop is now OCP-compliant.
        # It doesn't know or care what kind of handlers it has.
        for handler in self.handlers:
            try:
                # All handlers now have the same, simple signature
                handler.send_alert(alert)
            except Exception as e:
                logger.error(
                    f"Alerter handler {type(handler).__name__} failed: {e}",
                    exc_info=True,
                )
