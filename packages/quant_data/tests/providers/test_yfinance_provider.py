"""PR 2a-2: the Yahoo adapter returns raw (as-traded) bars and raw dividends.

``yf`` is replaced by a fake that serves what yfinance 1.7.0 returns: daily
``download`` frames with a tz-naive date index, intraday frames tz-aware in the
exchange zone, and ``Ticker.splits`` / ``.dividends`` tz-aware at the open, all
split-adjusted retroactively.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from quant_core.enums import CorporateActionType
from quant_core.interfaces import CorporateActionProvider, RateLimitedError
from quant_core.models import MarketDataRequest
from quant_data.providers import yfinance_provider
from quant_data.providers.yfinance_provider import YFinanceMarketDataProvider

NY = ZoneInfo("America/New_York")
BA = ZoneInfo("America/Argentina/Buenos_Aires")

# Yahoo's view of NVDA around the 10:1 split (ex 2024-06-10): all ÷10 before it.
DAILY = pd.DataFrame(
    {
        "Open": [121.0, 121.0, 121.0],
        "High": [122.0, 122.0, 122.0],
        "Low": [120.0, 120.0, 120.0],
        "Close": [121.0, 121.79, 120.91],
        "Adj Close": [120.65, 121.44, 120.58],
        "Volume": [664_696_000, 313_434_100, 222_551_200],
    },
    index=pd.to_datetime(["2024-06-06", "2024-06-10", "2024-06-11"]),
)
INTRADAY = pd.DataFrame(
    {
        "Open": [121.0, 121.0],
        "High": [122.0, 122.0],
        "Low": [120.0, 120.0],
        "Close": [121.0, 121.79],
        "Adj Close": [121.0, 121.79],
        "Volume": [1_000_000, 1_000_000],
    },
    index=pd.DatetimeIndex(
        [
            datetime(2024, 6, 7, 15, 30, tzinfo=NY),
            datetime(2024, 6, 10, 9, 30, tzinfo=NY),
        ]
    ),
)
SPLITS = pd.Series(
    [10.0], index=pd.DatetimeIndex([datetime(2024, 6, 10, 9, 30, tzinfo=NY)])
)
DIVIDENDS = pd.Series(
    [0.004, 0.01],
    index=pd.DatetimeIndex(
        [
            datetime(2024, 3, 5, 9, 30, tzinfo=NY),
            datetime(2024, 6, 11, 9, 30, tzinfo=NY),
        ]
    ),
)


class FakeYF:
    def __init__(self, frames=None, splits=SPLITS, dividends=DIVIDENDS):
        self.frames = frames or {"1d": DAILY, "1h": INTRADAY}
        self.splits, self.dividends = splits, dividends
        self.tickers_created = 0
        self.downloads = []

    def download(self, symbol, start, end, interval, auto_adjust, progress, **kw):
        assert auto_adjust is False
        self.downloads.append((symbol, interval))
        return self.frames[interval]

    def Ticker(self, symbol):
        self.tickers_created += 1
        return SimpleNamespace(splits=self.splits, dividends=self.dividends)


@pytest.fixture
def fake(monkeypatch):
    fake = FakeYF()
    monkeypatch.setattr(yfinance_provider, "yf", fake)
    return fake


def _request(interval, start=datetime(2024, 6, 1, tzinfo=timezone.utc)):
    return MarketDataRequest(
        instrument_id="NVDA",
        interval=interval,
        start_date=start,
        end_date=datetime(2024, 6, 12, tzinfo=timezone.utc),
    )


def test_daily_bars_are_raw(fake):
    bars = YFinanceMarketDataProvider().get_bars(_request("1d"))

    by_day = {b.timestamp.date().isoformat(): b for b in bars}
    pre = by_day["2024-06-06"]
    assert pre.close == pytest.approx(1210.0)
    assert pre.open == pytest.approx(1210.0)
    assert pre.volume == pytest.approx(66_469_600)
    assert pre.dollar_volume == pytest.approx(1210.0 * 66_469_600)
    assert by_day["2024-06-10"].close == pytest.approx(121.79)
    # Daily bars keep the local-date-at-midnight-UTC key PR 2 established.
    assert pre.timestamp == datetime(2024, 6, 6, tzinfo=timezone.utc)


def test_a_session_without_prices_is_dropped(fake):
    """Yahoo sends all-NaN rows (or a NaN volume) for some BYMA sessions; one
    must not fail the whole download (a missing value is dropped, never filled)."""
    nan = float("nan")
    gap = pd.DataFrame(
        {c: [nan] for c in DAILY.columns}, index=pd.to_datetime(["2024-06-07"])
    )
    no_volume = DAILY.iloc[[0]].assign(Volume=nan)
    no_volume.index = pd.to_datetime(["2024-06-12"])
    fake.frames["1d"] = pd.concat([DAILY, gap, no_volume]).sort_index()

    bars = YFinanceMarketDataProvider().get_bars(_request("1d"))

    assert [b.timestamp.date().isoformat() for b in bars] == [
        "2024-06-06",
        "2024-06-10",
        "2024-06-11",
    ]


def test_intraday_bars_are_raw_too(fake):
    bars = YFinanceMarketDataProvider().get_bars(_request("1h"))

    assert bars[0].close == pytest.approx(1210.0)  # 2024-06-07, pre-split
    assert bars[0].volume == pytest.approx(100_000)
    assert bars[1].close == pytest.approx(121.79)  # the ex-date open
    assert bars[0].timestamp == datetime(2024, 6, 7, 19, 30, tzinfo=timezone.utc)


def test_split_history_is_fetched_once_per_symbol(fake):
    provider = YFinanceMarketDataProvider()
    provider.get_bars(_request("1d"))
    provider.get_bars(_request("1h"))
    provider.get_corporate_actions("NVDA")
    assert fake.tickers_created == 1


def test_corporate_actions_are_raw_and_keyed_by_local_date(fake):
    provider = YFinanceMarketDataProvider()
    assert isinstance(provider, CorporateActionProvider)

    actions = provider.get_corporate_actions("NVDA")

    splits = [a for a in actions if a.action_type is CorporateActionType.SPLIT]
    dividends = [a for a in actions if a.action_type is CorporateActionType.DIVIDEND]
    assert [(s.ex_date, s.ratio) for s in splits] == [
        (datetime(2024, 6, 10, tzinfo=timezone.utc), 10.0)
    ]
    assert [(d.ex_date, d.amount) for d in dividends] == [
        (datetime(2024, 3, 5, tzinfo=timezone.utc), pytest.approx(0.04)),
        (datetime(2024, 6, 11, tzinfo=timezone.utc), pytest.approx(0.01)),
    ]
    assert all(a.instrument_id == "NVDA" for a in actions)


def test_byma_ex_date_keeps_its_local_date(monkeypatch):
    split = pd.Series(
        [10.0], index=pd.DatetimeIndex([datetime(2026, 8, 3, 11, 0, tzinfo=BA)])
    )
    monkeypatch.setattr(
        yfinance_provider, "yf", FakeYF(splits=split, dividends=pd.Series(dtype=float))
    )
    (action,) = YFinanceMarketDataProvider().get_corporate_actions("YPFD.BA")
    assert action.ex_date == datetime(2026, 8, 3, tzinfo=timezone.utc)


def test_symbol_without_actions(monkeypatch):
    monkeypatch.setattr(
        yfinance_provider,
        "yf",
        FakeYF(splits=pd.Series(dtype=float), dividends=pd.Series(dtype=float)),
    )
    provider = YFinanceMarketDataProvider()
    assert provider.get_corporate_actions("GGAL.BA") == []
    assert provider.get_bars(_request("1d"))[0].close == pytest.approx(121.0)


def test_yahoo_rate_limit_becomes_the_shared_error(monkeypatch):
    import yfinance

    class Limited(FakeYF):
        def download(self, *a, **kw):
            raise yfinance.exceptions.YFRateLimitError()

        def Ticker(self, symbol):
            raise yfinance.exceptions.YFRateLimitError()

    monkeypatch.setattr(yfinance_provider, "yf", Limited())
    provider = YFinanceMarketDataProvider()
    with pytest.raises(RateLimitedError):
        provider.get_bars(_request("1d"))
    with pytest.raises(RateLimitedError):
        provider.get_corporate_actions("NVDA")


def test_declares_a_conservative_rate_limit():
    limit = YFinanceMarketDataProvider.rate_limit
    assert limit is not None
    assert limit.min_interval_s >= 1.0 and limit.max_retries >= 1
    assert YFinanceMarketDataProvider().source_name == "YFinanceMarketDataProvider"
