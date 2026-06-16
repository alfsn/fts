# tests/unit/test_resampling.py

from datetime import datetime, timedelta, timezone

import pytest

from trading_bot.core.enums import BarType
from trading_bot.core.schemas import BarData
from trading_bot.utils.resampling import resample_bars, timeframe_to_seconds


def test_timeframe_to_seconds():
    assert timeframe_to_seconds("1m") == 60
    assert timeframe_to_seconds("5m") == 300
    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_seconds("1d") == 86400
    assert timeframe_to_seconds("1w") == 604800

    with pytest.raises(ValueError):
        timeframe_to_seconds("")
    with pytest.raises(ValueError):
        timeframe_to_seconds("abc")
    with pytest.raises(ValueError):
        timeframe_to_seconds("1x")


def test_resample_bars_empty():
    assert resample_bars([], "1h") == []


def test_resample_bars_1m_to_5m():
    # Create 5 consecutive 1-minute bars
    start_time = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    bars = []

    # Prices:
    # Bar 0 (12:00): O=10, H=12, L=9,  C=11, Vol=100
    # Bar 1 (12:01): O=11, H=13, L=10, C=12, Vol=150
    # Bar 2 (12:02): O=12, H=14, L=11, C=13, Vol=200
    # Bar 3 (12:03): O=13, H=15, L=12, C=14, Vol=250
    # Bar 4 (12:04): O=14, H=16, L=13, C=15, Vol=300

    for i in range(5):
        bars.append(
            BarData(
                timestamp=start_time + timedelta(minutes=i),
                open=10.0 + i,
                high=12.0 + i,
                low=9.0 + i,
                close=11.0 + i,
                volume=100.0 + i * 50,
                bar_type=BarType.TIME,
                interval="1m",
                ticks_count=2,
                dollar_volume=(11.0 + i) * (100.0 + i * 50),
            )
        )

    resampled = resample_bars(bars, "5m")
    assert len(resampled) == 1

    res_bar = resampled[0]
    assert res_bar.timestamp == start_time
    assert res_bar.open == 10.0
    assert res_bar.high == 16.0
    assert res_bar.low == 9.0
    assert res_bar.close == 15.0
    assert res_bar.volume == 1000.0  # 100 + 150 + 200 + 250 + 300
    assert res_bar.ticks_count == 10  # 2 * 5
    assert res_bar.interval == "5m"
    assert res_bar.bar_type == BarType.TIME


def test_resample_bars_invalid_upsample():
    # Trying to resample 1-hour bars to 5-minute bars should fail
    start_time = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    bars = [
        BarData(
            timestamp=start_time,
            open=10.0,
            high=12.0,
            low=9.0,
            close=11.0,
            volume=100.0,
            bar_type=BarType.TIME,
            interval="1h",
            ticks_count=20,
            dollar_volume=1100.0,
        )
    ]

    with pytest.raises(ValueError, match="cannot be a higher frequency"):
        resample_bars(bars, "5m")
