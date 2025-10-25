"""
Abstract Base Classes for the Monitoring & Alerting (Module 5).

This file defines the abstract interface for any alerting service,
allowing the bot to send notifications to different platforms
(e.g., Telegram, Discord, Email) without changing the core logic.
"""

from abc import ABC, abstractmethod

from ...core.schemas import Alert


class BaseAlerter(ABC):
    """
    Abstract base class for an alerting service.

    Defines the interface for sending a formatted alert to an
    external service.
    """

    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """
        Sends an alert to the implemented service.

        :param alert: The Alert object containing the message
                      and severity.
        :return: True if the alert was sent successfully,
                 False otherwise.
        """
        pass
