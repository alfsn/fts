from datetime import datetime, timezone

from quant_core.enums import CorporateActionType
from quant_data.db.models import CorporateActionRecord


def test_corporate_action_record_creation(db_session):
    record = CorporateActionRecord(
        source="test",
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
