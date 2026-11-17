import pytest
from mock_providers import MockIBKRProvider, MockTiingoProvider
from quant_core.enums import BarType
from quant_core.models import BarData
from quant_data.db.models import Base
from quant_data.providers import TieredDataProvider
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


def test_rate_limited_tier_falls_through_after_retries(db_session):
    """A tier that keeps answering 429 is retried `max_retries` times through
    the throttle, then the waterfall moves on. No real sleeping."""
    from quant_core.interfaces import RateLimitedError
    from quant_core.models import MarketDataRequest, RateLimit
    from quant_data.providers.throttle import Throttle

    class _Limited(MockIBKRProvider):
        rate_limit = RateLimit(min_interval_s=1.0, max_retries=2, backoff_s=10.0)
        calls = 0

        def get_bars(self, request):
            type(self).calls += 1
            raise RateLimitedError("MockIBKR")

    sleeps = []
    clock = {"now": 0.0}

    def sleep(s):
        sleeps.append(s)
        clock["now"] += s

    tiered = TieredDataProvider(
        db_session,
        [_Limited(), MockTiingoProvider()],
        throttle=Throttle(clock=lambda: clock["now"], sleep=sleep),
    )
    bars = tiered.get_bars(MarketDataRequest(instrument_id="MSFT", interval="1d"))

    assert len(bars) == 1  # served by tier 2
    assert _Limited.calls == 3
    assert sleeps == [10.0, 20.0]


def test_create_default_is_gone():
    assert not hasattr(TieredDataProvider, "create_default")


# --- PR 2a-3: cache rules (D4 single source, D9 gaps, revisions, D8 actions) ---


def _seed(db_session, symbol, days, source, close=1.0):
    """Puts daily bars straight into the cache, as an earlier run would have."""
    from datetime import datetime, timezone

    from quant_data.db.repositories import MarketDataRepository

    repo = MarketDataRepository(db_session)
    bars = [_bar(datetime(2021, 1, d, tzinfo=timezone.utc)) for d in days]
    for b in bars:
        b.close = close
    repo.save_bars(symbol, bars, source=source)
    db_session.commit()


def _day(d, month=1):
    from datetime import datetime, timezone

    return datetime(2021, month, d, tzinfo=timezone.utc)


def test_bars_are_cached_with_the_providers_source(db_session):
    from quant_data.db.repositories import MarketDataRepository

    provider = _RecordingProvider()
    TieredDataProvider(db_session, [provider]).get_bars(
        _range_request(_day(1), _day(5))
    )

    assert (
        MarketDataRepository(db_session).bar_source("AAPL", "1d")
        == provider.source_name
    )


def test_provider_switch_refetches_the_full_range_and_never_stitches(db_session):
    _seed(db_session, "AAPL", range(1, 11), source="OldVendor", close=5.0)
    provider = _RecordingProvider()  # answers the top-up with a different source
    tiered = TieredDataProvider(db_session, [provider])

    bars = tiered.get_bars(_range_request(_day(1), _day(20)))

    assert provider.requests[-1].start_date == _day(1)  # full range, not the tail
    assert len(bars) == 20
    assert {b.close for b in bars} == {1.0}  # nothing left from OldVendor
    from quant_data.db.repositories import MarketDataRepository

    assert (
        MarketDataRepository(db_session).bar_source("AAPL", "1d")
        == provider.source_name
    )


def test_failed_full_refetch_after_a_switch_serves_the_old_cache(db_session):
    _seed(db_session, "AAPL", range(1, 11), source="OldVendor", close=5.0)

    class _TailOnly(_RecordingProvider):
        def get_bars(self, request):
            if request.start_date == _day(1):
                raise ValueError("outage on the big request")
            return super().get_bars(request)

    bars = TieredDataProvider(db_session, [_TailOnly()]).get_bars(
        _range_request(_day(1), _day(20))
    )
    assert len(bars) == 10 and {b.close for b in bars} == {5.0}


def test_interior_gap_triggers_exactly_one_full_refetch(db_session):
    provider = _RecordingProvider()
    _seed(
        db_session,
        "AAPL",
        list(range(1, 11)) + list(range(20, 32)),
        source=provider.source_name,
    )
    tiered = TieredDataProvider(db_session, [provider])

    bars = tiered.get_bars(_range_request(_day(1), _day(31)))

    assert [(r.start_date, r.end_date) for r in provider.requests] == [
        (_day(1), _day(31))
    ]
    assert len(bars) == 31  # the hole is filled


def test_persistent_gap_is_served_with_a_warning(db_session, caplog):
    class _Gappy(_RecordingProvider):
        def get_bars(self, request):
            return [
                b for b in super().get_bars(request) if not 10 < b.timestamp.day < 20
            ]

    provider = _Gappy()
    _seed(
        db_session,
        "AAPL",
        list(range(1, 11)) + list(range(20, 32)),
        source=provider.source_name,
    )
    tiered = TieredDataProvider(db_session, [provider])

    with caplog.at_level("WARNING"):
        bars = tiered.get_bars(_range_request(_day(1), _day(31)))

    assert len(provider.requests) == 1
    assert len(bars) == 22
    assert "gap" in caplog.text.lower()


def test_small_gaps_do_not_refetch(db_session):
    """A long weekend or an Argentine bridge holiday (<= 5 days) is normal."""
    provider = _RecordingProvider()
    _seed(
        db_session, "AAPL", [1, 2, 3, 8, 9, 10], source=provider.source_name
    )  # 3 -> 8 is 5 days
    tiered = TieredDataProvider(db_session, [provider])

    tiered.get_bars(_range_request(_day(1), _day(10)))

    assert provider.requests[-1].start_date == _day(10)  # only the usual tail top-up


def test_revised_last_bar_is_updated(db_session):
    provider = _RecordingProvider()
    tiered = TieredDataProvider(db_session, [provider])
    tiered.get_bars(_range_request(_day(1), _day(5)))

    original = provider.get_bars

    def revised(request):
        bars = original(request)
        bars[0].close = 2.0  # the vendor corrected the last cached bar
        return bars

    provider.get_bars = revised
    bars = tiered.get_bars(_range_request(_day(1), _day(6)))

    assert bars[4].timestamp == _day(5) and bars[4].close == 2.0


class _ActionsProvider(_RecordingProvider):
    """Also a CorporateActionProvider; `actions` is what it answers, or raises."""

    def __init__(self, actions=None):
        super().__init__()
        self.actions = actions if actions is not None else []
        self.action_calls = 0

    def get_corporate_actions(self, instrument_id):
        self.action_calls += 1
        if isinstance(self.actions, Exception):
            raise self.actions
        return self.actions


def _split(symbol="AAPL"):
    from quant_core.enums import CorporateActionType
    from quant_core.models import CorporateAction

    return CorporateAction(
        instrument_id=symbol,
        action_type=CorporateActionType.SPLIT,
        ex_date=_day(4),
        ratio=4.0,
    )


def _make_action_provider():
    from quant_core.interfaces import CorporateActionProvider

    class _P(_ActionsProvider, CorporateActionProvider):
        pass

    return _P


def test_corporate_actions_are_refreshed_every_run_and_cached(db_session):
    P = _make_action_provider()
    provider = P([_split()])
    tiered = TieredDataProvider(db_session, [provider])
    from quant_core.interfaces import CorporateActionProvider

    assert isinstance(tiered, CorporateActionProvider)

    first = tiered.get_corporate_actions("AAPL")
    second = tiered.get_corporate_actions("AAPL")

    assert first == second == [_split()]
    assert provider.action_calls == 2  # D8: every run
    from quant_data.db.repositories import MarketDataRepository

    assert MarketDataRepository(db_session).load_corporate_actions("AAPL") == [_split()]


def test_corporate_actions_are_served_from_cache_when_the_vendor_fails(db_session):
    P = _make_action_provider()
    provider = P([_split()])
    tiered = TieredDataProvider(db_session, [provider])
    tiered.get_corporate_actions("AAPL")

    provider.actions = ValueError("vendor down")
    assert tiered.get_corporate_actions("AAPL") == [_split()]


def test_corporate_actions_come_from_the_bars_source(db_session):
    """Bars cached from vendor A: actions are asked from A, not from the first tier."""
    P = _make_action_provider()

    class VendorA(P):
        pass

    class VendorB(P):
        pass

    a, b = VendorA([_split()]), VendorB([])
    _seed(db_session, "AAPL", [1, 2], source="VendorA")
    tiered = TieredDataProvider(db_session, [b, a])

    assert tiered.get_corporate_actions("AAPL") == [_split()]
    assert (b.action_calls, a.action_calls) == (0, 1)


def test_corporate_actions_without_any_provider_are_empty(db_session):
    tiered = TieredDataProvider(db_session, [_RecordingProvider()])  # bars only
    assert tiered.get_corporate_actions("BTC/USDT") == []
