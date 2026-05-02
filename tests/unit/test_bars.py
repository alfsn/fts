# tests/unit/test_bars.py

from datetime import datetime, timedelta, timezone

import pytest

from trading_bot.core.enums import BarType, OrderSide
from trading_bot.core.schemas import Trade
from trading_bot.data_ingestion.bars import BarFactory


@pytest.fixture
def base_time():
    return datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


def test_time_bar_aggregator(base_time):
    agg = BarFactory.create_aggregator(
        BarType.TIME, "MKT1", interval=timedelta(minutes=5)
    )

    # Trade 1 at 10:01
    t1 = Trade(
        price=100.0,
        size=10.0,
        timestamp=base_time + timedelta(minutes=1),
        side=OrderSide.BUY,
    )
    bar = agg.add_trade(t1)
    assert bar is None
    assert agg.current_bar.open == 100.0
    assert agg.next_bar_boundary == base_time + timedelta(minutes=5)

    # Trade 2 at 10:04
    t2 = Trade(
        price=105.0,
        size=5.0,
        timestamp=base_time + timedelta(minutes=4),
        side=OrderSide.BUY,
    )
    bar = agg.add_trade(t2)
    assert bar is None
    assert agg.current_bar.high == 105.0
    assert agg.current_bar.volume == 15.0

    # Trade 3 at 10:06 (should trigger completion of 10:05 bar)
    t3 = Trade(
        price=102.0,
        size=2.0,
        timestamp=base_time + timedelta(minutes=6),
        side=OrderSide.SELL,
    )
    bar = agg.add_trade(t3)

    assert bar is not None
    assert bar.close == 105.0
    assert bar.volume == 15.0
    assert bar.timestamp == base_time + timedelta(minutes=5)

    # New bar should be started with t3
    assert agg.current_bar.open == 102.0
    assert agg.next_bar_boundary == base_time + timedelta(minutes=10)


def test_volume_bar_aggregator(base_time):
    agg = BarFactory.create_aggregator(BarType.VOLUME, "MKT1", threshold=100.0)

    # Trade 1: 60 volume
    t1 = Trade(price=10.0, size=60.0, timestamp=base_time, side=OrderSide.BUY)
    bar = agg.add_trade(t1)
    assert bar is None

    # Trade 2: 50 volume (crosses 100)
    t2 = Trade(
        price=11.0,
        size=50.0,
        timestamp=base_time + timedelta(seconds=10),
        side=OrderSide.BUY,
    )
    bar = agg.add_trade(t2)

    assert bar is not None
    assert bar.volume == 110.0
    assert bar.open == 10.0
    assert bar.close == 11.0
    assert bar.timestamp == base_time + timedelta(seconds=10)
    assert agg.current_bar is None


def test_dollar_bar_aggregator(base_time):
    agg = BarFactory.create_aggregator(BarType.DOLLAR, "MKT1", threshold=1000.0)

    # Trade 1: 10 * 60 = 600 dollars
    t1 = Trade(price=10.0, size=60.0, timestamp=base_time, side=OrderSide.BUY)
    bar = agg.add_trade(t1)
    assert bar is None

    # Trade 2: 20 * 30 = 600 dollars (total 1200, crosses 1000)
    t2 = Trade(
        price=20.0,
        size=30.0,
        timestamp=base_time + timedelta(seconds=10),
        side=OrderSide.BUY,
    )
    bar = agg.add_trade(t2)

    assert bar is not None
    assert bar.dollar_volume == 1200.0
    assert bar.timestamp == base_time + timedelta(seconds=10)
    assert agg.current_bar is None
