from datetime import datetime, timedelta, timezone

from quant_core.enums import CorporateActionType, Currency, RateType
from quant_data.db.models import CorporateActionRecord, FXRateRecord


def test_get_fx_rate(db_session, repo):
    now = datetime.now(timezone.utc)

    # Add rate 1 day ago
    r1 = FXRateRecord(
        base_currency=Currency.USD,
        quote_currency=Currency.ARS,
        rate_type=RateType.MEP,
        timestamp=now - timedelta(days=1),
        rate=1000.0,
    )
    # Add rate today
    r2 = FXRateRecord(
        base_currency=Currency.USD,
        quote_currency=Currency.ARS,
        rate_type=RateType.MEP,
        timestamp=now,
        rate=1100.0,
    )
    db_session.add_all([r1, r2])
    db_session.commit()

    # Query today -> should get r2
    rate = repo.get_fx_rate(Currency.USD, Currency.ARS, RateType.MEP, now)
    assert rate == 1100.0

    # Query 12 hours ago -> should get r1
    rate_past = repo.get_fx_rate(
        Currency.USD, Currency.ARS, RateType.MEP, now - timedelta(hours=12)
    )
    assert rate_past == 1000.0

    # Query missing rate -> None
    rate_missing = repo.get_fx_rate(Currency.EUR, Currency.USD, RateType.OFFICIAL, now)
    assert rate_missing is None


def test_get_corporate_actions(db_session, repo):
    now = datetime.now(timezone.utc)

    a1 = CorporateActionRecord(
        source="test",
        instrument_id="AAPL",
        action_type=CorporateActionType.SPLIT,
        ex_date=now - timedelta(days=10),
        ratio=4.0,
    )
    a2 = CorporateActionRecord(
        source="test",
        instrument_id="AAPL",
        action_type=CorporateActionType.DIVIDEND,
        ex_date=now - timedelta(days=5),
        amount=0.5,
    )
    a3 = CorporateActionRecord(
        source="test",
        instrument_id="MSFT",
        action_type=CorporateActionType.DIVIDEND,
        ex_date=now,
        amount=1.0,
    )
    db_session.add_all([a1, a2, a3])
    db_session.commit()

    # Get all for AAPL
    actions_aapl = repo.get_corporate_actions("AAPL")
    assert len(actions_aapl) == 2

    # Filter by date
    actions_recent = repo.get_corporate_actions(
        "AAPL", start_date=now - timedelta(days=7)
    )
    assert len(actions_recent) == 1
    assert actions_recent[0].action_type == CorporateActionType.DIVIDEND


# --- PR 2a-3: provenance, upserts, corporate-action cache -----------------------


def _bar(ts, close, interval="1d"):
    from quant_core.enums import BarType
    from quant_core.models import BarData

    return BarData(
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        bar_type=BarType.TIME,
        interval=interval,
        ticks_count=1,
        dollar_volume=close,
    )


def test_save_bars_records_the_source_and_updates_revised_rows(repo):
    day = datetime(2024, 1, 2, tzinfo=timezone.utc)

    assert repo.save_bars("AAPL", [_bar(day, 100.0)], source="yahoo") == 1
    # A late correction to an already cached bar is applied, not skipped.
    assert repo.save_bars("AAPL", [_bar(day, 101.0)], source="yahoo") == 0

    (bar,) = repo.load_bars("AAPL", "1d")
    assert bar.close == 101.0
    assert repo.bar_source("AAPL", "1d") == "yahoo"
    assert repo.bar_source("AAPL", "1h") is None


def test_save_bars_refuses_to_mix_sources(repo):
    day = datetime(2024, 1, 2, tzinfo=timezone.utc)
    repo.save_bars("AAPL", [_bar(day, 100.0)], source="yahoo")

    import pytest

    with pytest.raises(ValueError, match="yahoo"):
        repo.save_bars("AAPL", [_bar(day + timedelta(days=1), 1.0)], source="ibkr")


def test_replace_bars_swaps_the_whole_series_for_that_interval(repo):
    day = datetime(2024, 1, 2, tzinfo=timezone.utc)
    repo.save_bars(
        "AAPL", [_bar(day, 100.0), _bar(day + timedelta(days=1), 101.0)], source="yahoo"
    )
    repo.save_bars("AAPL", [_bar(day, 100.0, "1h")], source="yahoo")

    repo.replace_bars(
        "AAPL", "1d", [_bar(day + timedelta(days=1), 99.0)], source="ibkr"
    )

    assert [b.close for b in repo.load_bars("AAPL", "1d")] == [99.0]
    assert repo.bar_source("AAPL", "1d") == "ibkr"
    assert len(repo.load_bars("AAPL", "1h")) == 1  # other intervals untouched


def test_save_corporate_actions_upserts_and_reads_back(repo):
    from quant_core.models import CorporateAction

    split = CorporateAction(
        instrument_id="NVDA",
        action_type=CorporateActionType.SPLIT,
        ex_date=datetime(2024, 6, 10, tzinfo=timezone.utc),
        ratio=10.0,
    )
    dividend = CorporateAction(
        instrument_id="NVDA",
        action_type=CorporateActionType.DIVIDEND,
        ex_date=datetime(2024, 6, 11, tzinfo=timezone.utc),
        amount=0.01,
    )
    repo.save_corporate_actions("NVDA", [split, dividend], source="yahoo")
    revised = dividend.model_copy(update={"amount": 0.02})
    repo.save_corporate_actions("NVDA", [split, revised], source="yahoo")

    actions = repo.load_corporate_actions("NVDA")
    assert [type(a) for a in actions] == [CorporateAction, CorporateAction]
    assert [(a.action_type, a.amount, a.ratio) for a in actions] == [
        (CorporateActionType.SPLIT, None, 10.0),
        (CorporateActionType.DIVIDEND, 0.02, None),
    ]
    assert actions[0].ex_date == datetime(2024, 6, 10, tzinfo=timezone.utc)
    assert {r.source for r in repo.get_corporate_actions("NVDA")} == {"yahoo"}


def test_save_corporate_actions_from_a_new_source_replaces_the_old(repo):
    from quant_core.models import CorporateAction

    old = CorporateAction(
        instrument_id="NVDA",
        action_type=CorporateActionType.SPLIT,
        ex_date=datetime(2021, 7, 20, tzinfo=timezone.utc),
        ratio=4.0,
    )
    repo.save_corporate_actions("NVDA", [old], source="yahoo")
    repo.save_corporate_actions("NVDA", [], source="ibkr")
    assert repo.load_corporate_actions("NVDA") == []


def test_load_bars_keeps_intervals_apart_and_compares_bounds_in_utc(repo):
    """M2a: one series per interval, and an aware bound in another zone is
    converted to UTC before SQLite compares it with the stored naive UTC."""
    noon = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)
    repo.save_bars("BTC", [_bar(noon, 1.0, "1h")], source="ccxt")
    repo.save_bars("BTC", [_bar(noon, 2.0, "1d")], source="ccxt")

    plus_three = timezone(timedelta(hours=3))
    # 14:00 at +03:00 is 11:00 UTC: before noon UTC, so the bar is included.
    bars = repo.load_bars(
        "BTC", "1h", start=datetime(2024, 1, 2, 14, tzinfo=plus_three)
    )

    assert [b.close for b in bars] == [1.0]
    assert bars[0].timestamp == noon  # aware UTC on the way out
