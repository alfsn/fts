# tests/unit/test_nets_plugin.py

from datetime import datetime

import numpy as np
import pytest
from nets.classifiers import DynamicThresholdClassifier, SimpleThresholdClassifier
from nets.enums import PredictionSignal

from trading_bot.core.enums import BarType
from trading_bot.core.schemas import BarData
from trading_bot.core.transforms import LogReturnTransform


def test_log_return_transform():
    transform = LogReturnTransform()
    prices = np.array([100.0, 110.0, 105.0]).reshape(-1, 1)
    returns = transform.transform(prices)

    # ln(110/100) = 0.0953
    # ln(105/110) = -0.0465
    assert len(returns) == 2
    assert pytest.approx(float(returns[0, 0]), 0.001) == 0.0953
    assert pytest.approx(float(returns[1, 0]), 0.001) == -0.0465


def test_classifiers():
    dummy = SimpleThresholdClassifier(threshold=0.01)
    assert dummy.classify(0.005, []) == PredictionSignal.FLAT
    assert dummy.classify(0.015, []) == PredictionSignal.UP
    assert dummy.classify(-0.015, []) == PredictionSignal.DOWN

    dynamic = DynamicThresholdClassifier(k=1.0, period=2)
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
    assert dynamic.classify(0.02, bars) == PredictionSignal.FLAT
    assert dynamic.classify(0.04, bars) == PredictionSignal.UP
    assert dynamic.classify(-0.04, bars) == PredictionSignal.DOWN
