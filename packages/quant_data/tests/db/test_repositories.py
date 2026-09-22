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
        instrument_id="AAPL",
        action_type=CorporateActionType.SPLIT,
        ex_date=now - timedelta(days=10),
        ratio=4.0,
    )
    a2 = CorporateActionRecord(
        instrument_id="AAPL",
        action_type=CorporateActionType.DIVIDEND,
        ex_date=now - timedelta(days=5),
        amount=0.5,
    )
    a3 = CorporateActionRecord(
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
