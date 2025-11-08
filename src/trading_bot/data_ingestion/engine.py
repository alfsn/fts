# src/trading_bot/data_ingestion/engine.py

import logging
from datetime import datetime, timezone
from typing import Dict, List

from ..core.schemas import ExternalData, IngestionEngineOutput, MarketData
from .abc import BaseExternalDataProvider, BaseMarketDataProvider

# Get a logger for this module
logger = logging.getLogger(__name__)


class DataIngestionEngine:
    """
    Orchestrates all data providers to fetch and assemble data.

    This class adheres to the Single Responsibility Principle by
    delegating all data *fetching* logic to the injected provider
    abstractions. Its only responsibility is to *coordinate* these
    providers and assemble their data into the `IngestionEngineOutput`
    data contract.

    This design follows the Dependency Inversion Principle, as it
    depends on abstractions (ABCs) rather than concrete implementations.
    """

    def __init__(
        self,
        market_provider: BaseMarketDataProvider,
        external_providers: List[BaseExternalDataProvider],
        market_ids: List[str],
    ):
        """
        Initializes the engine using Dependency Injection.

        :param market_provider: A concrete implementation of
                                BaseMarketDataProvider (e.g., PolymarketClient).
        :param external_providers: A list of concrete external data
                                   providers (e.g., TwitterSentimentProvider).
        :param market_ids: The specific list of market IDs this engine
                           should poll from the `market_provider`.
        """
        self.market_provider = market_provider
        self.external_providers = external_providers
        self.market_ids = market_ids
        logger.info(
            f"DataIngestionEngine initialized with {len(self.market_ids)} markets "
            f"and {len(self.external_providers)} external providers."
        )

    def fetch_all_data(self) -> IngestionEngineOutput:
        """
        Fetches data from all configured providers for one "tick".

        This method orchestrates the data fetching by:
        1. Polling the BaseMarketDataProvider for data on all `market_ids`.
        2. Polling all BaseExternalDataProviders.
        3. Assembling the results into a single IngestionEngineOutput object.

        This process is designed to be robust; a failure in fetching data
        for one market or from one external provider will be logged but
        will not stop the engine from processing the others.

        :return: An IngestionEngineOutput object for this tick.
        """
        # 1. Get timestamp
        timestamp = datetime.now(timezone.utc)

        # 2. Fetch all market data
        market_data_map: Dict[str, MarketData] = self._fetch_market_data()

        # 3. Fetch all external data
        all_external_data: List[ExternalData] = self._fetch_external_data()

        # 4. Assemble and return
        output = IngestionEngineOutput(
            timestamp=timestamp,
            market_data=market_data_map,
            external_data=all_external_data,
        )

        logger.debug(
            f"Data tick generated: {len(output.market_data)} markets, "
            f"{len(output.external_data)} external events."
        )
        return output

    def _fetch_market_data(self) -> Dict[str, MarketData]:
        """
        Helper method to poll the market provider for all market IDs.
        """
        market_data_map: Dict[str, MarketData] = {}
        for market_id in self.market_ids:
            try:
                market_data = self.market_provider.get_market_data(market_id)
                if market_data:
                    market_data_map[market_id] = market_data
                else:
                    logger.warning(f"No market data returned for {market_id}")
            except Exception as e:
                # Log the error but continue the loop
                logger.error(
                    f"Failed to fetch market data for {market_id}: {e}",
                    exc_info=True,  # Includes stack trace in log
                )
        return market_data_map

    def _fetch_external_data(self) -> List[ExternalData]:
        """
        Helper method to poll all external data providers.
        """
        all_external_data: List[ExternalData] = []
        for provider in self.external_providers:
            try:
                external_data = provider.fetch_data()
                all_external_data.extend(external_data)
            except Exception as e:
                # Log the error but continue the loop
                logger.error(
                    f"Failed to fetch from external provider "
                    f"'{provider.source_name}': {e}",
                    exc_info=True,
                )
        return all_external_data
