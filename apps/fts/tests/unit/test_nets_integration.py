from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
from quant_core.enums import AssetType, BarType, Geography, SignalType
from quant_core.models import TradFiDetails
from trading_bot.config import ComponentConfig, PluginLoader
from trading_bot.core.schemas import (
    BarData,
    IngestionEngineOutput,
    Instrument,
    MarketData,
    OrderBook,
    SizingInput,
)

# tests/unit/test_nets_integration.py


...


def test_plugin_component_loading():
    # 1. Test loading NetsStrategy
    mock_predictor = MagicMock()
    strategy_config = ComponentConfig(
        class_path="nets.strategies.nets_strategy.NetsStrategy",
        params={
            "predictor": mock_predictor,
            "transform": PluginLoader.instantiate(
                ComponentConfig(
                    class_path="trading_bot.core.transforms.LogReturnTransform"
                )
            ),
            "output_selector": PluginLoader.instantiate(
                ComponentConfig(
                    class_path="nets.output_selectors.SimpleThresholdClassifier",
                    params={"threshold": 0.001},
                )
            ),
            "lookback_period": 5,
        },
    )
    strategy = PluginLoader.instantiate(strategy_config)
    assert strategy.name == "nets_strategy_v1"

    # Set mock return for predictor
    mock_predictor.predict.return_value = np.array([0.01])

    # 2. Test evaluate
    bars = [
        BarData(
            timestamp=datetime.now(),
            open=100 + i,
            high=101 + i,
            low=99 + i,
            close=100.1 + i,
            volume=100,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        )
        for i in range(10)
    ]
    market_data = MarketData(
        instrument_id="GGAL",
        order_book=OrderBook(bids=[], asks=[]),
        recent_trades=[],
        details=Instrument(
            instrument_id="GGAL",
            name="GGAL",
            details=TradFiDetails(
                asset_type=AssetType.STOCK,
                geography=Geography.US,
                industry="Tech",
                currency="USD",
            ),
        ),
        recent_bars=bars,
    )
    tick_data = IngestionEngineOutput(
        timestamp=datetime.now(),
        market_data={"GGAL": market_data},
        external_data=[],
        bars={"GGAL": bars},
    )

    signals = strategy.evaluate(tick_data)
    assert len(signals) == 1
    assert signals[0].instrument_id == "GGAL"


def test_confidence_sizer_integration():
    sizer_config = ComponentConfig(
        class_path="nets.sizing.confidence_sizer.ConfidenceSizer",
        params={"base_amount_quote": 1000.0},
    )
    sizer = PluginLoader.instantiate(sizer_config)

    # Mock input
    instrument_id = "GGAL"
    signal = PluginLoader.instantiate(
        ComponentConfig(
            class_path="trading_bot.core.schemas.TradeSignal",
            params={
                "instrument_id": instrument_id,
                "strategy_name": "test",
                "signal_type": SignalType.BUY,
                "confidence": 0.8,
            },
        )
    )

    bars = [
        BarData(
            timestamp=datetime.now(),
            open=100,
            high=101,
            low=99,
            close=100.0,
            volume=100,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        )
    ]
    market_data = MarketData(
        instrument_id=instrument_id,
        order_book=OrderBook(bids=[], asks=[]),
        recent_trades=[],
        details=Instrument(
            instrument_id=instrument_id,
            name="GGAL",
            details=TradFiDetails(
                asset_type=AssetType.STOCK,
                geography=Geography.US,
                industry="Tech",
                currency="USD",
            ),
        ),
        recent_bars=bars,
    )

    from quant_core.models import CashBalance, Portfolio

    portfolio_state = Portfolio(
        cash_balances=[CashBalance(currency="USD", total=10000.0, available=5000.0)],
        positions=[],
        open_orders=[],
    )

    sizing_input = SizingInput(
        signal=signal, market_data=market_data, portfolio_state=portfolio_state
    )

    output = sizer.calculate_size(sizing_input)
    # 1000 * 0.8 = 800 USD
    # 800 / 100 (price) = 8 shares
    assert output.amount_quote == 800.0
    assert output.quantity_shares == 8.0
