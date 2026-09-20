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

    def _get_execution_price(self, input_data: SizingInput) -> float:
        """
        Helper method to retrieve the execution price for the trade,
        crossing the spread from the order book if available, or
        falling back to the last close bar price.

        Returns 0.0 if no price can be determined.
        """
        import logging

        from ..core.enums import SignalType

        logger = logging.getLogger(self.__class__.__module__)
        market_data = input_data.market_data
        signal = input_data.signal
        book = market_data.order_book

        if book is None or (not book.bids and not book.asks):
            if market_data.recent_bars:
                price = market_data.recent_bars[-1].close
                logger.debug(
                    f"No order book for {signal.market_id}. "
                    f"Falling back to last bar close price: {price}"
                )
                return price
            else:
                logger.warning(
                    f"No order book or historical bars available for {signal.market_id}."
                )
                return 0.0

        try:
            if signal.signal_type == SignalType.BUY:
                if not book.asks:
                    logger.warning(f"No asks available for {signal.market_id}.")
                    return 0.0
                return book.asks[0].price
            else:  # SignalType.SELL
                if not book.bids:
                    logger.warning(f"No bids available for {signal.market_id}.")
                    return 0.0
                return book.bids[0].price
        except IndexError:
            logger.error(f"Order book was empty for {signal.market_id} despite check.")
            return 0.0
