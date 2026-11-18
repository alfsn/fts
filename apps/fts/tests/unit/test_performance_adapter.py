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


def _equity_log(days, strategy_beta=0.5, seed=1):
    """Daily equity of a strategy holding `strategy_beta` of BTC, plus noise."""
    import numpy as np

    rng = np.random.default_rng(seed)
    btc = 30000 * np.cumprod(np.r_[1.0, 1 + rng.normal(0.001, 0.03, days - 1)])
    btc_r = np.r_[0.0, btc[1:] / btc[:-1] - 1]
    equity = 1000 * np.cumprod(1 + strategy_beta * btc_r + rng.normal(0, 0.002, days))
    return [
        SimpleNamespace(timestamp=T0 + timedelta(days=i), equity=e, close=c)
        for i, (e, c) in enumerate(zip(equity, btc))
    ]


def test_benchmark_metrics_against_buy_and_hold():
    from trading_bot.core.performance import build_benchmark_metrics

    m = build_benchmark_metrics(_equity_log(400))
    assert m.n_obs == 399
    assert m.beta == pytest.approx(0.5, abs=0.03)
    assert m.beta_ci_low < 0.5 < m.beta_ci_high


def test_benchmark_metrics_short_backtest_and_unpriced_ticks_are_nan():
    from trading_bot.core.performance import build_benchmark_metrics

    log = _equity_log(400)
    log[5].close = 0.0  # the engine logs 0.0 when it has no bar
    m = build_benchmark_metrics(log)
    assert m.n_obs == 398  # the unpriced day is dropped, not a -100% day

    short = build_benchmark_metrics(_equity_log(60))
    assert short.n_obs == 59 and short.beta != short.beta  # NaN below 252
