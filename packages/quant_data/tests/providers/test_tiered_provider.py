import pytest
from quant_core.enums import BarType
from quant_core.models import BarData
from quant_data.db.models import Base
from quant_data.providers import (
    MockIBKRProvider,
    MockTiingoProvider,
    TieredDataProvider,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_tiered_provider_fallback(db_session):
    from quant_core.models import MarketDataRequest

    # Tier 1 fails for AAPL and VXX
    tier1 = MockIBKRProvider(failing_tickers={"AAPL", "VXX"})

    # Tier 2 fails for VXX
    tier2 = MockTiingoProvider(failing_tickers={"VXX"})

    # Build engine
    engine = TieredDataProvider(db_session, [tier1, tier2])

    # 1. MSFT should succeed on Tier 1
    req1 = MarketDataRequest(instrument_id="MSFT", interval="1d")
    msft_bars = engine.get_bars(req1)
    assert len(msft_bars) == 1
    # Check it didn't fail

    # 2. AAPL should fail on Tier 1 and succeed on Tier 2
    req2 = MarketDataRequest(instrument_id="AAPL", interval="1d")
    aapl_bars = engine.get_bars(req2)
    assert len(aapl_bars) == 1

    # 3. VXX should fail on Tier 1 and Tier 2, and since there is no Tier 3, it should raise ValueError
    with pytest.raises(ValueError, match="All providers in the waterfall failed"):
        req3 = MarketDataRequest(instrument_id="VXX", interval="1d")
        engine.get_bars(req3)
