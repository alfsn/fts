# tests/test_strategy_universe.py

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

# Import Schemas
from trading_bot.core.schemas import (
    MarketData,
    MarketDetails,
    OrderBook,
    PriceLevel,
)

# Import the class we are testing
from trading_bot.strategy.universe import UniverseBuilder

# --- Pytest Fixtures ---

NOW = datetime.now(timezone.utc)


@pytest.fixture
def base_builder() -> UniverseBuilder:
    """A UniverseBuilder with default settings."""
    return UniverseBuilder(
        min_days_to_expiry=1,
        max_days_to_expiry=30,
        min_liquidity_usd=100,
        max_spread_bps=500,  # 5%
    )


def create_mock_market_data(
    market_id: str,
    name: str,
    end_date: datetime,
    bids: List[PriceLevel],
    asks: List[PriceLevel],
) -> MarketData:
    """Helper to create MarketData fixtures."""
    return MarketData(
        market_id=market_id,
        order_book=OrderBook(bids=bids, asks=asks),
        recent_trades=[],
        details=MarketDetails(
            market_id=market_id,
            name=name,
            end_date=end_date,
            resolution_source="test",
        ),
    )


@pytest.fixture
def market_perfect() -> MarketData:
    """A market that should pass all default filters."""
    return create_mock_market_data(
        market_id="PERFECT",
        name="Will this market pass?",
        end_date=NOW + timedelta(days=15),
        bids=[PriceLevel(price=0.50, size=1000)],  # 500 USD
        asks=[PriceLevel(price=0.51, size=1000)],  # 510 USD
    )


@pytest.fixture
def market_expired() -> MarketData:
    """A market that has already expired."""
    return create_mock_market_data(
        market_id="EXPIRED",
        name="Expired Market",
        end_date=NOW - timedelta(days=1),
        bids=[PriceLevel(price=0.50, size=1000)],
        asks=[PriceLevel(price=0.51, size=1000)],
    )


@pytest.fixture
def market_too_far() -> MarketData:
    """A market that expires too far in the future."""
    return create_mock_market_data(
        market_id="TOO_FAR",
        name="Market 2099",
        end_date=NOW + timedelta(days=100),
        bids=[PriceLevel(price=0.50, size=1000)],
        asks=[PriceLevel(price=0.51, size=1000)],
    )


@pytest.fixture
def market_illiquid_bid() -> MarketData:
    """A market with insufficient liquidity on the bid side."""
    return create_mock_market_data(
        market_id="ILLIQUID_BID",
        name="Illiquid Market",
        end_date=NOW + timedelta(days=15),
        bids=[PriceLevel(price=0.50, size=10)],  # 5 USD
        asks=[PriceLevel(price=0.51, size=1000)],
    )


@pytest.fixture
def market_wide_spread() -> MarketData:
    """A market with a spread wider than the 5% (500bps) limit."""
    return create_mock_market_data(
        market_id="WIDE_SPREAD",
        name="Wide Spread Market",
        end_date=NOW + timedelta(days=15),
        bids=[PriceLevel(price=0.50, size=1000)],
        asks=[PriceLevel(price=0.53, size=1000)],  # 6% spread
    )


@pytest.fixture
def market_no_bids() -> MarketData:
    """A market with no bids."""
    return create_mock_market_data(
        market_id="NO_BIDS",
        name="No Bids Market",
        end_date=NOW + timedelta(days=15),
        bids=[],
        asks=[PriceLevel(price=0.51, size=1000)],
    )


@pytest.fixture
def market_include_kw() -> MarketData:
    """A market with a keyword to be included."""
    return create_mock_market_data(
        market_id="INCLUDE_KW",
        name="Market about [TRUMP]",
        end_date=NOW + timedelta(days=15),
        bids=[PriceLevel(price=0.50, size=1000)],
        asks=[PriceLevel(price=0.51, size=1000)],
    )


@pytest.fixture
def market_exclude_kw() -> MarketData:
    """A market with a keyword to be excluded."""
    return create_mock_market_data(
        market_id="EXCLUDE_KW",
        name="Market about [BIDEN]",
        end_date=NOW + timedelta(days=15),
        bids=[PriceLevel(price=0.50, size=1000)],
        asks=[PriceLevel(price=0.51, size=1000)],
    )


# --- Test Cases ---


class TestUniverseBuilder:
    """Tests the UniverseBuilder filtering logic."""

    def test_filter_by_expiry(
        self,
        base_builder: UniverseBuilder,
        market_perfect: MarketData,
        market_expired: MarketData,
        market_too_far: MarketData,
    ):
        """Tests the expiry date filter logic directly."""
        assert base_builder._filter_by_expiry(market_perfect, NOW) is True
        assert base_builder._filter_by_expiry(market_expired, NOW) is False
        assert base_builder._filter_by_expiry(market_too_far, NOW) is False

    def test_filter_by_name(
        self,
        base_builder: UniverseBuilder,
        market_include_kw: MarketData,
        market_exclude_kw: MarketData,
    ):
        """Tests the name filter logic directly."""
        # 1. No filters (default)
        assert base_builder._filter_by_name(market_include_kw) is True
        assert base_builder._filter_by_name(market_exclude_kw) is True

        # 2. Include filter
        builder_include = UniverseBuilder(name_keywords_include=["trump"])
        assert builder_include._filter_by_name(market_include_kw) is True
        assert builder_include._filter_by_name(market_exclude_kw) is False

        # 3. Exclude filter
        builder_exclude = UniverseBuilder(name_keywords_exclude=["biden"])
        assert builder_exclude._filter_by_name(market_include_kw) is True
        assert builder_exclude._filter_by_name(market_exclude_kw) is False

        # 4. Both filters
        builder_both = UniverseBuilder(
            name_keywords_include=["trump"], name_keywords_exclude=["biden"]
        )
        assert builder_both._filter_by_name(market_include_kw) is True
        assert builder_both._filter_by_name(market_exclude_kw) is False

    def test_filter_by_liquidity_and_spread(
        self,
        base_builder: UniverseBuilder,
        market_perfect: MarketData,
        market_illiquid_bid: MarketData,
        market_wide_spread: MarketData,
        market_no_bids: MarketData,
    ):
        """Tests the liquidity and spread filter logic directly."""
        assert base_builder._filter_by_liquidity_and_spread(market_perfect) is True
        assert (
            base_builder._filter_by_liquidity_and_spread(market_illiquid_bid) is False
        )
        assert base_builder._filter_by_liquidity_and_spread(market_wide_spread) is False
        assert base_builder._filter_by_liquidity_and_spread(market_no_bids) is False

    def test_build_universe_integration_default(
        self,
        base_builder: UniverseBuilder,
        market_perfect,
        market_expired,
        market_illiquid_bid,
        market_wide_spread,
        market_no_bids,
    ):
        """
        Tests the main `build_universe` method with a default builder
        to ensure only the perfect market passes.
        """
        all_data = {
            m.market_id: m
            for m in [
                market_perfect,
                market_expired,
                market_illiquid_bid,
                market_wide_spread,
                market_no_bids,
            ]
        }
        universe = base_builder.build_universe(all_data)
        assert universe == ["PERFECT"]

    def test_build_universe_integration_with_keywords(
        self,
        market_perfect,
        market_include_kw,
        market_exclude_kw,
    ):
        """
        Tests the main `build_universe` method with a builder configured
        for keyword filtering.
        """
        # Builder that only wants "TRUMP" and excludes "BIDEN"
        builder_keywords = UniverseBuilder(
            min_liquidity_usd=100,
            max_spread_bps=500,
            name_keywords_include=["trump"],
            name_keywords_exclude=["biden"],
        )

        all_data = {
            m.market_id: m
            for m in [market_perfect, market_include_kw, market_exclude_kw]
        }

        universe = builder_keywords.build_universe(all_data)

        # "PERFECT" is filtered out (doesn't have "trump")
        # "EXCLUDE_KW" is filtered out (has "biden")
        assert universe == ["INCLUDE_KW"]
