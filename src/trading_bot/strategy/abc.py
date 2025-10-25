"""
Abstract Base Classes for the Strategy Engine (Module 2).

This file defines the core interface for any trading strategy.
A strategy's role is to receive data and produce signals.
"""

from abc import ABC, abstractmethod
from typing import List

from ...core.schemas import IngestionEngineOutput, TradeSignal


class BaseStrategy(ABC):
    """
    Abstract base class for a trading strategy.

    Defines the core interface for the Strategy Engine. A strategy's
    role is to receive comprehensive data from the Ingestion Engine
    and produce one or more trade signals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        A unique name for the strategy (e.g., 'kelly_momentum_v1').
        This is used for logging and populating the
        TradeSignal.strategy_name field.

        :return: The strategy's name as a string.
        """
        pass

    @abstractmethod
    def evaluate(self, data: IngestionEngineOutput) -> List[TradeSignal]:
        """
        The core logic method. This method is called by the Strategy
        Engine on each 'tick' (e.g., new data packet, or time
        interval) with the latest available data.

        The strategy should analyze the market and external data to
        generate a list of trading signals.

        :param data: The IngestionEngineOutput object containing all
                     the latest market and external data.
        :return: A list of TradeSignal objects. Can be an empty list
                 if the strategy decides to do nothing.
        """
        pass
