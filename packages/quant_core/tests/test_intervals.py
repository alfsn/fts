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
