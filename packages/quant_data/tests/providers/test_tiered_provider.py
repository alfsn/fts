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


class _RecordingProvider(MockIBKRProvider):
    """Serves one bar per day of the requested range and records each request."""

    def __init__(self):
        super().__init__()
        self.requests = []

    def get_bars(self, request):
        from datetime import timedelta

        self.requests.append(request)
        start, end = request.resolve_bounds()
        day, bars = start, []
        while day <= end:
            bars.append(_bar(day))
            day += timedelta(days=1)
        return bars


def _bar(ts):
    return BarData(
        timestamp=ts,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=0.0,
        bar_type=BarType.TIME,
        interval="1d",
        ticks_count=1,
        dollar_volume=0.0,
    )


def _range_request(start, end):
    from quant_core.models import MarketDataRequest

    return MarketDataRequest(
        instrument_id="AAPL", interval="1d", start_date=start, end_date=end
    )


def test_range_request_is_not_truncated_by_count(db_session):
    from datetime import datetime, timezone

    provider = _RecordingProvider()
    tiered = TieredDataProvider(db_session, [provider])
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end = datetime(2021, 12, 31, tzinfo=timezone.utc)

    bars = tiered.get_bars(_range_request(start, end))

    assert len(bars) == 365  # default count=100 must not apply to a range
    assert all(isinstance(b, BarData) for b in bars)
    assert bars[0].timestamp.tzinfo is not None


def test_full_cache_tops_up_only_the_tail(db_session):
    from datetime import datetime, timezone

    provider = _RecordingProvider()
    tiered = TieredDataProvider(db_session, [provider])
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    tiered.get_bars(_range_request(start, datetime(2021, 6, 30, tzinfo=timezone.utc)))

    bars = tiered.get_bars(
        _range_request(start, datetime(2021, 7, 10, tzinfo=timezone.utc))
    )

    top_up = provider.requests[-1]
    assert top_up.start_date.date() == datetime(2021, 6, 30).date()
    assert len(bars) == 191
    assert all(isinstance(b, BarData) for b in bars)


def test_cache_not_reaching_start_refetches_full_range(db_session):
    from datetime import datetime, timezone

    provider = _RecordingProvider()
    tiered = TieredDataProvider(db_session, [provider])
    end = datetime(2021, 12, 31, tzinfo=timezone.utc)
    tiered.get_bars(_range_request(datetime(2021, 6, 1, tzinfo=timezone.utc), end))

    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    bars = tiered.get_bars(_range_request(start, end))

    assert provider.requests[-1].start_date == start
    assert len(bars) == 365


def test_failed_top_up_serves_cache(db_session):
    from datetime import datetime, timezone

    provider = _RecordingProvider()
    tiered = TieredDataProvider(db_session, [provider])
    request = _range_request(
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 31, tzinfo=timezone.utc),
    )
    tiered.get_bars(request)
    provider.get_bars = lambda request: []  # e.g. holiday or provider outage

    assert len(tiered.get_bars(request)) == 31
