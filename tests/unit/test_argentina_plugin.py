# tests/unit/test_argentina_plugin.py

from unittest.mock import MagicMock

import pytest
from argentina.data_providers import CCLProvider

from trading_bot.core.schemas import OrderBook, PriceLevel


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
    # CCL = (250050 * 10) / 50.05 = 49960.03996
    assert len(data) == 1
    assert pytest.approx(data[0].content["ccl_rate"], 0.1) == 49960.0
