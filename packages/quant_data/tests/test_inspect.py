"""The read-only cache inspector (W3b)."""

from datetime import datetime, timedelta, timezone

import pytest
from quant_core.enums import BarType, CorporateActionType
from quant_core.models import BarData, CorporateAction
from quant_data import inspect
from quant_data.db import MarketDataRepository, create_db_session, init_cache
from quant_data.providers import TieredDataProvider
from quant_data.reference.gaps import GAP_RULES

DAY0 = datetime(2026, 7, 27, 20, tzinfo=timezone.utc)
SPLIT_DAY = datetime(2026, 8, 3, 13, tzinfo=timezone.utc)


def _bar(ts, close):
    return BarData(
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        bar_type=BarType.TIME,
        interval="1d",
        ticks_count=1,
        dollar_volume=close,
    )


def _seed(url, closes, symbol="YPFD.BA", split=None, days=None):
    with create_db_session(url) as session:
        init_cache(session.get_bind())
        repo = MarketDataRepository(session)
        days = days or [DAY0 + timedelta(days=i) for i in range(len(closes))]
        repo.save_bars(
            symbol,
            [_bar(d, c) for d, c in zip(days, closes)],
            "YFinanceMarketDataProvider",
        )
        if split:
            repo.save_corporate_actions(
                symbol,
                [
                    CorporateAction(
                        instrument_id=symbol,
                        action_type=CorporateActionType.SPLIT,
                        ex_date=SPLIT_DAY,
                        ratio=split,
                    )
                ],
                "YFinanceMarketDataProvider",
            )
        session.commit()


@pytest.fixture
def url(tmp_path):
    return f"sqlite:///{tmp_path / 'cache.db'}"


def test_a_split_breaks_the_raw_series_and_not_the_adjusted_one(url):
    # 10:1 on 2026-08-03: raw drops from 100 to 10, split-adjusted is flat.
    _seed(url, [100.0] * 7 + [10.0] * 3, split=10.0)
    (row,) = inspect.cached_series(url)
    assert (row.instrument_id, row.interval, row.bars) == ("YPFD.BA", "1d", 10)
    view = inspect.inspect_series("YPFD.BA", "1d", url)
    assert view.raw.min() == 10.0 and view.raw.max() == 100.0
    assert view.split.round(9).nunique() == 1
    assert view.known_source and view.gaps.empty


def test_gaps_use_the_vendors_rule(url):
    days = [DAY0, DAY0 + timedelta(days=1), DAY0 + timedelta(days=9)]
    _seed(url, [1.0, 1.0, 1.0], days=days)
    view = inspect.inspect_series("YPFD.BA", "1d", url)
    assert view.max_gap == timedelta(days=5)  # exchange hours (SessionGap)
    assert len(view.gaps) == 1


def test_pairs_that_disagree_give_no_ccl(url):
    # GGAL implies 1000, YPF 2000 ARS per USD: 100% apart, so the day is NaN.
    _seed(url, [100.0], "GGAL.BA")
    _seed(url, [1.0], "GGAL")
    _seed(url, [200.0], "YPFD.BA")
    _seed(url, [1.0], "YPF")
    view = inspect.ccl_view(url)
    assert view.implied.iloc[0].tolist() == [1000.0, 2000.0]
    assert view.ccl.isna().all() and view.rejected_days == 1


def test_every_vendor_takes_its_gap_rule_from_the_table():
    # The inspector reads GAP_RULES by source name. Import every provider
    # module so a new file is seen; `is` fails a rule written inline. The
    # wrapper saves bars under its inner vendor's name, so it needs no entry.
    import importlib
    import pkgutil

    import quant_data.providers as pkg
    from quant_core.interfaces import BaseMarketDataProvider

    for module in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{module.name}")

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    vendors = {
        cls
        for cls in subclasses(BaseMarketDataProvider)
        if cls.__module__.startswith(pkg.__name__ + ".")
        and cls is not TieredDataProvider
    }
    assert len(vendors) >= 3
    for cls in vendors:
        assert cls.gap_rule is GAP_RULES.get(cls.__name__), cls.__name__


def test_volume_bars_are_left_out_of_the_series_list(url):
    # Volume bars have no interval; they used to crash the list (and, at a
    # shared interval, inflate the time-bar count).
    _seed(url, [1.0, 1.0])
    volume_bar = _bar(DAY0, 1.0).model_copy(
        update={"bar_type": BarType.VOLUME, "interval": None}
    )
    with create_db_session(url) as session:
        MarketDataRepository(session).save_bars(
            "YPFD.BA", [volume_bar], "YFinanceMarketDataProvider"
        )
        session.commit()
    (row,) = inspect.cached_series(url)
    assert (row.interval, row.bars) == ("1d", 2)
