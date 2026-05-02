# tests/unit/test_data_ingestion.py

import logging
from datetime import datetime, timezone
from typing import Dict, List

import pytest

# Import ABCs and Schemas
from trading_bot.core.schemas import (
    BarData,
    ExternalData,
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
    OrderBook,
    PriceLevel,
    Trade,
)
from trading_bot.data_ingestion.abc import (
    BaseExternalDataProvider,
    BaseMarketDataProvider,
)

# Import the class we are testing
from trading_bot.data_ingestion.engine import DataIngestionEngine

# --- Fake (Stub) Implementations for Testing ---

# We create "Fakes" that perfectly implement the ABCs
# This allows for predictable testing without real API calls.


class FakeMarketProvider(BaseMarketDataProvider):
    """
    A fake implementation of the market data provider for testing.
    It returns predictable data and can be configured to simulate failures.
    """

    def __init__(self, data: Dict[str, MarketData], markets_to_fail: List[str] = None):
        self.data_to_return = data
        self.markets_to_fail = markets_to_fail or []

    def get_market_data(self, market_id: str) -> MarketData | None:
        """Returns pre-canned data or simulates a failure."""
        if market_id in self.markets_to_fail:
            raise ValueError(f"Simulated failure for {market_id}")
        return self.data_to_return.get(market_id)

    # --- Other ABC methods (not used by the engine) ---
    def list_tradable_markets(self) -> List[MarketDetails]:
        return [md.details for md in self.data_to_return.values()]

    def get_market_details(self, market_id: str) -> MarketDetails:
        return self.data_to_return.get(market_id).details

    def get_order_book(self, market_id: str) -> OrderBook:
        return self.data_to_return.get(market_id).order_book

    def get_trade_history(self, market_id: str) -> List[Trade]:
        return self.data_to_return.get(market_id).recent_trades

    def get_bars(self, market_id: str, count: int = 100) -> List[BarData]:
        return getattr(self.data_to_return.get(market_id), "recent_bars", [])


class FakeExternalProvider(BaseExternalDataProvider):
    """
    A fake implementation of the external data provider for testing.
    """

    def __init__(
        self,
        name: str,
        data: List[ExternalData],
        should_fail: bool = False,
    ):
        self._name = name
        self.data_to_return = data
        self.should_fail = should_fail

    @property
    def source_name(self) -> str:
        """A unique name for the data source."""
        return self._name

    def fetch_data(self) -> List[ExternalData]:
        """Returns pre-canned data or simulates a failure."""
        if self.should_fail:
            raise ConnectionError(f"Simulated failure for {self.source_name}")
        return self.data_to_return


# --- Pytest Fixtures ---


@pytest.fixture
def sample_market_data_1() -> MarketData:
    """A sample MarketData object for MKT1."""
    return MarketData(
        market_id="MKT1",
        order_book=OrderBook(
            bids=[PriceLevel(price=0.49, size=100)],
            asks=[PriceLevel(price=0.51, size=100)],
        ),
        recent_trades=[],
        details=MarketDetails(
            market_id="MKT1",
            name="Test Market 1",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        ),
    )


@pytest.fixture
def sample_market_data_2() -> MarketData:
    """A sample MarketData object for MKT2."""
    return MarketData(
        market_id="MKT2",
        order_book=OrderBook(
            bids=[PriceLevel(price=0.69, size=50)],
            asks=[PriceLevel(price=0.71, size=50)],
        ),
        recent_trades=[],
        details=MarketDetails(
            market_id="MKT2",
            name="Test Market 2",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        ),
    )


@pytest.fixture
def sample_external_data_1() -> ExternalData:
    """A sample ExternalData object from a news source."""
    return ExternalData(
        source="news_api",
        timestamp=datetime.now(timezone.utc),
        content={"headline": "Test headline 1", "sentiment": 0.8},
    )


@pytest.fixture
def sample_external_data_2() -> ExternalData:
    """A sample ExternalData object from a twitter source."""
    return ExternalData(
        source="twitter_sentiment",
        timestamp=datetime.now(timezone.utc),
        content={"tweet": "Test tweet", "sentiment": -0.5},
    )


# --- Test Cases ---


def test_engine_initialization(sample_market_data_1, sample_external_data_1):
    """
    Tests that the engine correctly stores its injected dependencies.
    """
    # Given
    market_provider = FakeMarketProvider(data={"MKT1": sample_market_data_1})
    external_provider = FakeExternalProvider(name="test", data=[sample_external_data_1])
    market_ids = ["MKT1"]

    # When
    engine = DataIngestionEngine(
        market_provider=market_provider,
        external_providers=[external_provider],
        market_ids=market_ids,
    )

    # Then
    assert engine.market_provider == market_provider
    assert engine.external_providers == [external_provider]
    assert engine.market_ids == market_ids


def test_fetch_all_data_happy_path(sample_market_data_1, sample_external_data_1):
    """
    Tests the engine's main loop with one of each provider,
    expecting a successful data assembly.
    """
    # Given
    market_provider = FakeMarketProvider(data={"MKT1": sample_market_data_1})
    external_provider = FakeExternalProvider(
        name="news_api", data=[sample_external_data_1]
    )
    engine = DataIngestionEngine(
        market_provider=market_provider,
        external_providers=[external_provider],
        market_ids=["MKT1"],
    )

    # When
    start_time = datetime.now(timezone.utc).timestamp()
    result = engine.fetch_all_data()
    end_time = datetime.now(timezone.utc).timestamp()

    # Then
    assert isinstance(result, IngestionEngineOutput)
    assert result.timestamp.timestamp() == pytest.approx(
        start_time, abs=end_time - start_time + 0.1
    )

    # Check market data
    assert len(result.market_data) == 1
    assert "MKT1" in result.market_data
    assert result.market_data["MKT1"] == sample_market_data_1

    # Check external data
    assert len(result.external_data) == 1
    assert result.external_data[0] == sample_external_data_1
    assert result.external_data[0].source == "news_api"


def test_fetch_all_data_multiple_providers(
    sample_market_data_1,
    sample_market_data_2,
    sample_external_data_1,
    sample_external_data_2,
):
    """
    Tests that the engine correctly polls and assembles data from
    multiple markets and multiple external sources.
    """
    # Given
    market_provider = FakeMarketProvider(
        data={"MKT1": sample_market_data_1, "MKT2": sample_market_data_2}
    )
    external_provider_1 = FakeExternalProvider(
        name="news_api", data=[sample_external_data_1]
    )
    external_provider_2 = FakeExternalProvider(
        name="twitter_sentiment", data=[sample_external_data_2]
    )

    engine = DataIngestionEngine(
        market_provider=market_provider,
        external_providers=[external_provider_1, external_provider_2],
        market_ids=["MKT1", "MKT2"],
    )

    # When
    result = engine.fetch_all_data()

    # Then
    # Check market data
    assert len(result.market_data) == 2
    assert "MKT1" in result.market_data
    assert "MKT2" in result.market_data

    # Check external data
    assert len(result.external_data) == 2
    assert sample_external_data_1 in result.external_data
    assert sample_external_data_2 in result.external_data


def test_fetch_market_data_partial_failure(
    sample_market_data_1, sample_external_data_1, caplog
):
    """
    Tests that the engine is robust to a failure in *one* of the
    market data fetches. It should log the error but still return
    the data it *did* get.
    """
    # Given
    market_provider = FakeMarketProvider(
        data={"MKT1": sample_market_data_1},  # Has data for MKT1
        markets_to_fail=["MKT_FAIL"],  # Will fail on MKT_FAIL
    )
    external_provider = FakeExternalProvider(
        name="news_api", data=[sample_external_data_1]
    )

    engine = DataIngestionEngine(
        market_provider=market_provider,
        external_providers=[external_provider],
        market_ids=["MKT1", "MKT_FAIL"],  # We ask for both
    )

    # When
    with caplog.at_level(logging.ERROR):
        result = engine.fetch_all_data()

    # Then
    # Check market data (only MKT1 should be present)
    assert len(result.market_data) == 1
    assert "MKT1" in result.market_data
    assert "MKT_FAIL" not in result.market_data

    # Check external data (should be unaffected)
    assert len(result.external_data) == 1
    assert result.external_data[0] == sample_external_data_1

    # Check logs
    assert "Failed to fetch market data for MKT_FAIL" in caplog.text


def test_fetch_external_data_partial_failure(
    sample_market_data_1, sample_external_data_1, caplog
):
    """
    Tests that the engine is robust to a failure in *one* of the
    external providers.
    """
    # Given
    market_provider = FakeMarketProvider(data={"MKT1": sample_market_data_1})

    # This one will work
    external_provider_1 = FakeExternalProvider(
        name="news_api", data=[sample_external_data_1]
    )
    # This one will fail
    external_provider_2 = FakeExternalProvider(
        name="twitter", data=[], should_fail=True
    )

    engine = DataIngestionEngine(
        market_provider=market_provider,
        external_providers=[external_provider_1, external_provider_2],
        market_ids=["MKT1"],
    )

    # When
    with caplog.at_level(logging.ERROR):
        result = engine.fetch_all_data()

    # Then
    # Check market data (should be unaffected)
    assert len(result.market_data) == 1
    assert "MKT1" in result.market_data

    # Check external data (should only have data from the good provider)
    assert len(result.external_data) == 1
    assert result.external_data[0] == sample_external_data_1

    # Check logs
    assert "Failed to fetch from external provider 'twitter'" in caplog.text


def test_fetch_market_data_returns_none(sample_market_data_1, caplog):
    """
    Tests that the engine handles a provider returning None for a
    market_id instead of raising an error.
    """
    # Given
    # The provider has data for MKT1, but we will ask for MKT_NONE
    market_provider = FakeMarketProvider(data={"MKT1": sample_market_data_1})
    engine = DataIngestionEngine(
        market_provider=market_provider,
        external_providers=[],
        market_ids=["MKT_NONE"],
    )

    # When
    with caplog.at_level(logging.WARNING):
        result = engine.fetch_all_data()

    # Then
    assert len(result.market_data) == 0
    assert "No market data returned for MKT_NONE" in caplog.text
