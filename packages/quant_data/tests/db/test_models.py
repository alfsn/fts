from datetime import datetime, timezone

from quant_core.enums import CorporateActionType, Currency, RateType
from quant_data.db.models import CorporateActionRecord, FXRateRecord


def test_fx_rate_record_creation(db_session):
    record = FXRateRecord(
        base_currency=Currency.USD,
        quote_currency=Currency.ARS,
        rate_type=RateType.MEP,
        timestamp=datetime.now(timezone.utc),
        rate=1000.0,
    )
    db_session.add(record)
    db_session.commit()

    retrieved = db_session.query(FXRateRecord).first()
    assert retrieved is not None
    assert retrieved.rate == 1000.0
    assert retrieved.base_currency == Currency.USD
    assert retrieved.quote_currency == Currency.ARS


def test_corporate_action_record_creation(db_session):
    record = CorporateActionRecord(
        instrument_id="AAPL",
        action_type=CorporateActionType.SPLIT,
        ex_date=datetime.now(timezone.utc),
        ratio=4.0,
    )
    db_session.add(record)
    db_session.commit()

    retrieved = db_session.query(CorporateActionRecord).first()
    assert retrieved is not None
    assert retrieved.instrument_id == "AAPL"
    assert retrieved.action_type == CorporateActionType.SPLIT
    assert retrieved.ratio == 4.0
