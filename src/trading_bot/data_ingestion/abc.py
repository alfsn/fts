"""
Abstract Base Classes for the Data Ingestion Engine (Module 1).

This file defines the abstract interfaces for any data provider,
whether it fetches internal market data (like order books) or
external event-driven data (like API results).
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..core.schemas import (
    BarData,
    ExternalData,
    MarketData,
    MarketDetails,
    OrderBook,
    Trade,
)


class BaseMarketDataProvider(ABC):
    """
    Abstract base class for a market data provider.

    Defines the interface for connecting to an exchange's API
    (e.g., Polymarket, Kalshi) to fetch internal market data.
    A concrete implementation of this class will exist for each
    supported exchange.
    """

    @classmethod
    def from_args(cls, args: Any) -> "BaseMarketDataProvider":
        """
        Creates an instance of the provider from parsed command-line arguments.
        Subclasses should override this if they support command-line instantiation.
        """
        raise NotImplementedError(
            f"Provider '{cls.__name__}' does not support initialization from command-line arguments."
        )

    @abstractmethod
    def list_tradable_markets(self) -> Sequence[MarketDetails]:
        """
        Fetches a list of all available or tradable markets
        from the exchange.

        :return: A sequence of MarketDetails objects.
        """
        pass

    @abstractmethod
    def get_market_details(self, market_id: str) -> MarketDetails:
        """
        Fetches the static details for a single market (e.g.,
        question, resolution source, end date).

        :param market_id: The unique identifier for the market.
        :return: A MarketDetails object.
        """
        pass

    @abstractmethod
    def get_order_book(self, market_id: str) -> OrderBook:
        """
        Fetches the current order book for a specific market.

        :param market_id: The unique identifier for the market.
        :return: An OrderBook object.
        """
        pass

    @abstractmethod
    def get_trade_history(self, market_id: str) -> Sequence[Trade]:
        """
        Fetches the recent trade history for a specific market.

        :param market_id: The unique identifier for the market.
        :return: A sequence of Trade objects, typically sorted by time.
        """
        pass

    @abstractmethod
    def get_bars(self, market_id: str, count: int = 100) -> Sequence[BarData]:
        """
        Fetches the recent aggregated bars for a specific market.

        :param market_id: The unique identifier for the market.
        :param count: The number of recent bars to fetch.
        :return: A sequence of BarData objects.
        """
        pass

    @abstractmethod
    def get_market_data(self, market_id: str) -> MarketData:
        """
        Fetches a comprehensive snapshot of a single market.

        This method may internally call get_order_book, get_trade_history,
        and other methods to construct the final MarketData object.

        :param market_id: The unique identifier for the market.
        :return: A MarketData object.
        """
        pass


class BaseExternalDataProvider(ABC):
    """
    Abstract base class for an external data provider.

    Defines the interface for fetching real-world data (e.g., from
    a Twitter API, a weather service, or any other non-exchange source)
    that is used by a strategy to form a prediction.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        A unique name for the data source (e.g., 'twitter_sentiment').
        This will be populated in the ExternalData schema.

        :return: A string name for the data source.
        """
        pass

    @abstractmethod
    def fetch_data(self) -> Sequence[ExternalData]:
        """
        Fetches new external data points.

        The implementation should handle its own state (e.g., knowing
        what data it has already fetched) if necessary.

        :return: A sequence of ExternalData objects. Can be an empty
                 list if no new data is available.
        """
        pass
