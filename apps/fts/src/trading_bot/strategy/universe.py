# src/trading_bot/strategy/universe.py

"""
Defines the UniverseBuilder class.

This module provides logic for filtering and defining the "investable universe"
based on a set of configurable rules. It filters a large list of
potential markets down to a smaller list that the bot should
actively monitor and trade.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.schemas import MarketData, PriceLevel

# Get a logger for this module
logger = logging.getLogger(__name__)


class UniverseBuilder:
    """
    Applies a set of configurable filters to define the investable universe.

    This class is designed to be stateless and configurable. You can
    initialize it with your desired parameters (e.g., minimum liquidity,
    time to expiry) and then call `build_universe` with the latest
    market data to get a filtered list of market IDs.
    """

    def __init__(
        self,
        min_days_to_expiry: float = 1.0,
        max_days_to_expiry: float = 90.0,
        min_liquidity_usd: float = 1000.0,
        max_spread_bps: int = 500,  # 500 bps = 5%
        name_keywords_include: Optional[List[str]] = None,
        name_keywords_exclude: Optional[List[str]] = None,
    ):
        """
        Initializes the filter configuration.

        :param min_days_to_expiry: The minimum number of days before a
                                   market expires to be included.
        :param max_days_to_expiry: The maximum number of days before a
                                   market expires to be included.
        :param min_liquidity_usd: The minimum USD value required on the
                                  best bid AND best ask.
        :param max_spread_bps: The maximum allowed spread between the best
                               bid and ask in basis points (1% = 100 bps).
        :param name_keywords_include: A list of keywords (case-insensitive)
                                      where at least one must be in the
                                      market name. If None, this filter
                                      is skipped.
        :param name_keywords_exclude: A list of keywords (case-insensitive)
                                      that must NOT be in the market name.
                                      If None, this filter is skipped.
        """
        self.min_days_to_expiry = min_days_to_expiry
        self.max_days_to_expiry = max_days_to_expiry
        self.min_liquidity_usd = min_liquidity_usd
        self.max_spread_bps = max_spread_bps
        self.name_keywords_include = (
            [kw.lower() for kw in name_keywords_include]
            if name_keywords_include
            else None
        )
        self.name_keywords_exclude = (
            [kw.lower() for kw in name_keywords_exclude]
            if name_keywords_exclude
            else None
        )
        logger.info("UniverseBuilder initialized with filters.")

    def build_universe(self, all_market_data: Dict[str, MarketData]) -> List[str]:
        """
        Applies all configured filters to the provided market data.

        :param all_market_data: A dictionary mapping all available
                                market_ids to their corresponding
                                MarketData objects.
        :return: A list of market_ids that passed all filters.
        """
        now = datetime.now(timezone.utc)
        investable_markets: List[str] = []

        for market_id, data in all_market_data.items():
            # 1. Filter by Expiry Date
            if not self._filter_by_expiry(data, now):
                logger.debug(f"Market {market_id} excluded: Fails expiry filter.")
                continue

            # 2. Filter by Name (Keywords)
            if not self._filter_by_name(data):
                logger.debug(f"Market {market_id} excluded: Fails name keyword filter.")
                continue

            # 3. Filter by Liquidity and Spread
            if not self._filter_by_liquidity_and_spread(data):
                logger.debug(
                    f"Market {market_id} excluded: Fails liquidity/spread filter."
                )
                continue

            # If all filters passed:
            logger.debug(f"Market {market_id} included in universe.")
            investable_markets.append(market_id)

        logger.info(
            f"Universe built. {len(investable_markets)} / "
            f"{len(all_market_data)} markets included."
        )
        return investable_markets

    def _filter_by_expiry(self, data: MarketData, now: datetime) -> bool:
        """Checks if the market's end date is within the allowed range."""
        days_to_expiry = (data.details.end_date - now).total_seconds() / (60 * 60 * 24)
        return (
            self.min_days_to_expiry <= days_to_expiry
            and days_to_expiry <= self.max_days_to_expiry
        )

    def _filter_by_name(self, data: MarketData) -> bool:
        """Checks if the market name matches include/exclude keywords."""
        market_name_lower = data.details.name.lower()

        # Check for exclusions
        if self.name_keywords_exclude:
            if any(kw in market_name_lower for kw in self.name_keywords_exclude):
                return False  # Found an excluded word

        # Check for inclusions
        if self.name_keywords_include:
            if not any(kw in market_name_lower for kw in self.name_keywords_include):
                return False  # Did not find any of the required words

        return True  # Passed both checks

    def _filter_by_liquidity_and_spread(self, data: MarketData) -> bool:
        """
        Checks if the market has minimum liquidity and is within the
        max spread.
        """
        book = data.order_book
        if not book.bids or not book.asks:
            return False  # No liquidity on one or both sides

        best_bid: PriceLevel = book.bids[0]
        best_ask: PriceLevel = book.asks[0]

        # Check liquidity
        bid_liquidity_usd = best_bid.price * best_bid.size
        ask_liquidity_usd = best_ask.price * best_ask.size

        if (
            bid_liquidity_usd < self.min_liquidity_usd
            or ask_liquidity_usd < self.min_liquidity_usd
        ):
            return False  # Not enough liquidity

        # Check spread
        if best_bid.price <= 0:  # Avoid division by zero
            return False

        spread_bps = ((best_ask.price - best_bid.price) / best_bid.price) * 10000

        return spread_bps <= self.max_spread_bps
