"""
Abstract Base Classes for the Risk & Position Management (Module 3).

This file defines the interface for position sizing strategies.
"""

from abc import ABC, abstractmethod

from ..core.enums import SizingStrategyType
from ..core.schemas import SizingInput, SizingOutput


class BaseSizingStrategy(ABC):
    """
    Abstract base class for a position sizing strategy.

    Defines the interface for calculating "how much" to risk on a
    given trade signal, based on the signal's properties (like
    confidence) and the current portfolio state.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> SizingStrategyType:
        """
        Returns the type of sizing strategy this is
        (e.g., KELLY_CRITERION, FIXED_AMOUNT).

        :return: A SizingStrategyType enum member.
        """
        pass

    @abstractmethod
    def calculate_size(self, input_data: SizingInput) -> SizingOutput:
        """
        Calculates the exact trade size in shares and/or quote currency.

        The logic inside this method will implement a specific
        formula (e.g., Kelly Criterion, fixed percentage).

        :param input_data: A SizingInput object containing the
                         TradeSignal, current PortfolioState,
                         and relevant MarketData.
        :return: A SizingOutput object specifying the amount_quote
                 and/or size_shares for the trade. If the
                 strategy decides not to trade (e.g., risk limit
                 hit or zero confidence), it should return a
                 SizingOutput with amount_quote=0 and size_shares=0.
        """
        pass
