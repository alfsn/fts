from datetime import datetime, timedelta, timezone

from quant_core.enums import BarType
from quant_core.models import BarData
from quant_data.history import fetch_daily_closes


def _bar(ts, close):
    return BarData(
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=0.0,
        bar_type=BarType.TIME,
        interval="1d",
        ticks_count=1,
        dollar_volume=0.0,
    )


class _Provider:
    """BYMA-style stamps at 14:00 UTC, NYSE-style at 13:30 UTC; one bar per day."""

    def __init__(self):
        self.requests = []

    def get_bars(self, request):
        self.requests.append(request)
        if request.instrument_id == "BROKEN":
            raise ValueError("no data")
        hour, minute = (14, 0) if request.instrument_id.endswith(".BA") else (13, 30)
        day = request.start_date.replace(hour=hour, minute=minute)
        bars = []
        while day < request.end_date:
            bars.append(_bar(day, 1.0))
            day += timedelta(days=1)
        return bars


def test_aligns_venues_on_date_and_skips_failures():
    provider = _Provider()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 11, tzinfo=timezone.utc)

    closes = fetch_daily_closes(provider, ["GGAL", "GGAL.BA", "BROKEN"], start, end)

    assert list(closes.columns) == ["GGAL", "GGAL.BA"]
    assert len(closes) == 10
    assert not closes.isna().any().any()  # 13:30 and 14:00 stamps share a row
    assert (closes.index == closes.index.normalize()).all()


def test_never_requests_todays_partial_bar():
    provider = _Provider()
    now = datetime.now(timezone.utc)

    closes = fetch_daily_closes(provider, ["AAPL"], now - timedelta(days=5), now)

    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    assert provider.requests[0].end_date == today
    assert closes.index.max() < today


class _Actions:
    def get_corporate_actions(self, instrument_id):
        from quant_core.enums import CorporateActionType
        from quant_core.models import CorporateAction

        if instrument_id == "BROKEN":
            raise ValueError("vendor down")
        if instrument_id == "GGAL.BA":
            return []
        return [
            CorporateAction(
                instrument_id=instrument_id,
                action_type=CorporateActionType.SPLIT,
                ex_date=datetime(2024, 6, 10, tzinfo=timezone.utc),
                ratio=10.0,
            )
        ]


def test_fetch_corporate_actions_per_symbol_and_skips_failures():
    from quant_data.history import fetch_corporate_actions

    actions = fetch_corporate_actions(_Actions(), ["NVDA", "GGAL.BA", "BROKEN", "NVDA"])

    assert sorted(actions) == ["GGAL.BA", "NVDA"]  # a failure is absent, not empty
    assert actions["GGAL.BA"] == []
    assert actions["NVDA"][0].ratio == 10.0
