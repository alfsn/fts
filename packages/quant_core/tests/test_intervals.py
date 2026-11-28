from datetime import timedelta

import pytest
from quant_core.intervals import BarGap, SessionGap, interval_to_timedelta


def test_intervals_parse_every_vendor_spelling():
    assert interval_to_timedelta("30m") == timedelta(minutes=30)
    assert interval_to_timedelta("4h") == timedelta(hours=4)
    assert interval_to_timedelta("1wk") == interval_to_timedelta("1w")
    assert interval_to_timedelta("1M") == interval_to_timedelta("1mo")
    assert interval_to_timedelta("1s") == timedelta(seconds=1)
    assert interval_to_timedelta("1H") == timedelta(hours=1)
    assert interval_to_timedelta("1D") == timedelta(days=1)
    with pytest.raises(ValueError):
        interval_to_timedelta("1x")


def test_a_24_7_venue_counts_bars_and_an_exchange_never_counts_a_weekend():
    # Five hourly crypto bars missing is a gap; a 65-hour equity weekend in
    # hourly bars is not.
    assert BarGap().max_gap("1h") == timedelta(hours=5)
    assert SessionGap().max_gap("1h") == timedelta(days=5)
    assert SessionGap().max_gap("1w") == timedelta(weeks=5)


def test_find_gaps_keeps_only_holes_longer_than_the_limit():
    from datetime import datetime, timedelta, timezone

    from quant_core.intervals import find_gaps

    days = [
        datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=d)
        for d in (0, 1, 6, 12)
    ]
    gaps = find_gaps(days, timedelta(days=5))
    # 1 -> 6 is exactly the limit (not a gap); 6 -> 12 is one.
    assert list(gaps["start"]) == [days[2]] and list(gaps["end"]) == [days[3]]
    assert find_gaps(days[:1], timedelta(days=5)).empty


def test_find_gaps_accepts_mixed_utc_offsets():
    # A Postgres cache in a non-UTC session zone returns +01:00 and +02:00
    # across a DST change; pandas refuses those without utc=True.
    from datetime import datetime, timedelta, timezone

    from quant_core.intervals import find_gaps

    days = [
        datetime(2026, 10, 24, 12, tzinfo=timezone(timedelta(hours=2))),
        datetime(2026, 10, 26, 12, tzinfo=timezone(timedelta(hours=1))),
    ]
    assert find_gaps(days, timedelta(days=1)).iloc[0]["length"] == timedelta(
        days=2, hours=1
    )


def test_the_gap_comparison_has_one_home():
    """Only find_gaps compares a hole with max_gap (W3b); the cache refetch,
    ingest and the workbench call it."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    pattern = re.compile(r">\s*max_gap")
    hits = [
        str(p.relative_to(root))
        for folder in ("packages", "apps")
        for p in (root / folder).rglob("*.py")
        if "tests" not in p.parts and pattern.search(p.read_text())
    ]
    assert hits == ["packages/quant_core/src/quant_core/intervals.py"]
