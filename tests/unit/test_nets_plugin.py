# tests/unit/test_nets_plugin.py

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from nets.data_providers import CCLProvider
from nets.flat_buckets import DummyFlat, DynamicFlat

from trading_bot.core.enums import BarType
from trading_bot.core.schemas import BarData, OrderBook, PriceLevel
from trading_bot.core.transforms import LogReturnTransform


def test_ccl_provider_calculation():
    market_provider = MagicMock()

    # Mock Local (GGAL ARS)
    local_ob = OrderBook(
        bids=[PriceLevel(price=250000.0, size=1.0)],
        asks=[PriceLevel(price=250100.0, size=1.0)],
    )
    # Mock ADR (GGAL USD)
    adr_ob = OrderBook(
        bids=[PriceLevel(price=50.0, size=1.0)], asks=[PriceLevel(price=50.1, size=1.0)]
    )

    market_provider.get_order_book.side_effect = lambda m_id: (
        local_ob if "local" in m_id else adr_ob
    )

    provider = CCLProvider(market_provider, "ggal_local", "ggal_adr", ratio=10.0)
    data = provider.fetch_data()

    # local_mid = 250050
    # adr_mid = 50.05
    # CCL = (250050 * 10) / 50.05 = 50,000 ARS/USD (approx)

    assert len(data) == 1
    assert (
        pytest.approx(data[0].content["ccl_rate"], 0.1) == 49960.0
    )  # (250050*10)/50.05


def test_log_return_transform():
    transform = LogReturnTransform()
    prices = [100.0, 110.0, 105.0]
    returns = transform.transform(prices)

    # ln(110/100) = 0.0953
    # ln(105/110) = -0.0465
    assert len(returns) == 2
    assert pytest.approx(returns[0], 0.001) == 0.0953
    assert pytest.approx(returns[1], 0.001) == -0.0465


def test_flat_buckets():
    dummy = DummyFlat(threshold=0.01)
    assert dummy.is_flat(0.005, []) is True
    assert dummy.is_flat(0.015, []) is False

    dynamic = DynamicFlat(k=1.0, period=2)
    bars = [
        BarData(
            timestamp=datetime.now(),
            open=100,
            high=102,
            low=98,
            close=100,
            volume=1,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        ),
        BarData(
            timestamp=datetime.now(),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        ),
    ]
    # ATR_pct = ((4/100) + (2/100)) / 2 = 0.03
    # Threshold = 1.0 * 0.03 = 0.03
    assert dynamic.is_flat(0.02, bars) is True
    assert dynamic.is_flat(0.04, bars) is False
