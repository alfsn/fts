from quant_core.enums import AssetType, Geography
from quant_core.models import TradFiDetails

"""
Abstract Base Classes for the Data Ingestion Engine (Module 1).

This file defines the abstract interfaces for any data provider,
whether it fetches internal market data (like order books) or
external event-driven data (like API results).
"""

from abc import ABC, abstractmethod
from typing import Sequence

from quant_core.interfaces import BaseMarketDataProvider

from ..core.schemas import ExternalData


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
