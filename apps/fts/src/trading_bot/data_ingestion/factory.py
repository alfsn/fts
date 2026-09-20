# src/trading_bot/data_ingestion/factory.py

import logging
from typing import Dict, Type

from trading_bot.data_ingestion.abc import BaseMarketDataProvider

logger = logging.getLogger(__name__)


class MarketDataProviderRegistry:
    _registry: Dict[str, Type[BaseMarketDataProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseMarketDataProvider]) -> None:
        cls._registry[name.lower()] = provider_class
        logger.debug(f"Registered provider '{name}' in registry.")

    @classmethod
    def get_provider_class(cls, name: str) -> Type[BaseMarketDataProvider]:
        name_lower = name.lower()
        if name_lower not in cls._registry:
            cls._lazy_load_plugin(name_lower)

        if name_lower not in cls._registry:
            raise ValueError(f"Provider '{name}' is not registered or supported.")
        return cls._registry[name_lower]

    @classmethod
    def _lazy_load_plugin(cls, name: str) -> None:
        """Attempts to dynamically import and register a plugin by name."""
        try:
            if name == "yfinance":
                from yfinance_plugin.data_providers import YFinanceMarketDataProvider

                cls.register("yfinance", YFinanceMarketDataProvider)
            elif name == "ccxt":
                from ccxt_plugin.data_providers import CCXTMarketDataProvider

                cls.register("ccxt", CCXTMarketDataProvider)
        except ImportError as e:
            logger.critical(
                f"Failed to import plugin '{name}'. Make sure it is installed: {e}"
            )
            raise e
