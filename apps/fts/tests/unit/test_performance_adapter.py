from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from quant_core.enums import OrderSide
from trading_bot.core.performance import build_performance_metrics, trade_win_rate

T0 = datetime(2026, 1, 1)


def _fill(minute, side, qty, price, instrument="AAPL", outcome=None):
    return SimpleNamespace(
        fill_timestamp=T0 + timedelta(minutes=minute),
        side=side,
        quantity=qty,
        fill_price=price,
        instrument_id=instrument,
        outcome=outcome,
    )


def test_win_rate_counts_closed_round_trips_only():
    trades = [
        _fill(0, OrderSide.BUY, 1, 10),
        _fill(1, OrderSide.SELL, 1, 12),  # win
        _fill(2, OrderSide.BUY, 1, 12),
        _fill(3, OrderSide.SELL, 1, 11),  # loss
        _fill(4, OrderSide.BUY, 5, 11),  # still open: not a win
    ]
    assert trade_win_rate(trades) == pytest.approx(0.5)


def test_win_rate_does_not_net_across_books():
    trades = [
        _fill(0, OrderSide.BUY, 1, 0.4, outcome="YES"),
        _fill(1, OrderSide.SELL, 1, 0.6, outcome="NO"),
    ]
    assert trade_win_rate(trades) != trade_win_rate(trades)  # NaN: nothing closed


def test_build_performance_metrics():
    equity = [
        SimpleNamespace(timestamp=T0 + timedelta(days=i), equity=v)
        for i, v in enumerate([100.0, 110.0, 99.0, 120.0])
    ]
    m = build_performance_metrics(equity, [_fill(0, OrderSide.BUY, 1, 10)])
    assert m.total_return == pytest.approx(0.20)
    assert m.max_drawdown == pytest.approx(-0.10)
    assert m.total_trades == 1
