# tests/test_dashboard_renderers.py


import plotly.graph_objects as go

from trading_bot.core.schemas import BacktestDetailDTO
from trading_bot.dashboard.renderers import (
    render_backtest_comparison,
    render_equity_curve,
    render_prediction_scatter,
    render_trade_overlay,
)


def test_render_equity_curve():
    dto = BacktestDetailDTO(
        run_id="run_001",
        strategy_name="TestStrategy",
        market_id="AAPL",
        total_return=15.0,
        sharpe_ratio=1.8,
        max_drawdown=4.5,
        win_rate=60.0,
        total_trades=10,
        equity_curve=[
            {
                "timestamp": "2026-01-01T00:00:00",
                "cash": 100.0,
                "position": 0.0,
                "close": 10.0,
                "equity": 100.0,
            },
            {
                "timestamp": "2026-01-01T01:00:00",
                "cash": 115.0,
                "position": 0.0,
                "close": 11.5,
                "equity": 115.0,
            },
        ],
    )

    fig_abs = render_equity_curve(dto, normalize=False)
    assert isinstance(fig_abs, go.Figure)

    fig_norm = render_equity_curve(dto, normalize=True)
    assert isinstance(fig_norm, go.Figure)


def test_render_trade_overlay():
    equity_curve = [
        {"timestamp": "2026-01-01T00:00:00", "close": 10.0},
        {"timestamp": "2026-01-01T01:00:00", "close": 12.0},
    ]
    trades = [
        {
            "id": 1,
            "order_id": "ord_1",
            "market_id": "AAPL",
            "side": "BUY",
            "fill_size": 1.0,
            "fill_price": 10.0,
            "fill_timestamp": "2026-01-01T00:00:00",
        }
    ]

    fig = render_trade_overlay(equity_curve, trades)
    assert isinstance(fig, go.Figure)


def test_render_backtest_comparison():
    d1 = BacktestDetailDTO(
        run_id="run_001",
        strategy_name="StratA",
        equity_curve=[
            {"timestamp": "2026-01-01T00:00:00", "equity": 100.0},
            {"timestamp": "2026-01-01T01:00:00", "equity": 110.0},
        ],
    )
    d2 = BacktestDetailDTO(
        run_id="run_002",
        strategy_name="StratB",
        equity_curve=[
            {"timestamp": "2026-01-01T00:00:00", "equity": 100.0},
            {"timestamp": "2026-01-01T01:00:00", "equity": 105.0},
        ],
    )

    fig_cal = render_backtest_comparison([d1, d2], align_by_index=False)
    assert isinstance(fig_cal, go.Figure)

    fig_idx = render_backtest_comparison([d1, d2], align_by_index=True)
    assert isinstance(fig_idx, go.Figure)


def test_render_prediction_scatter():
    preds = [
        {"confidence": 0.85, "actual_future_return": 0.02, "predicted_signal": "BUY"},
        {"confidence": 0.70, "actual_future_return": -0.01, "predicted_signal": "SELL"},
    ]
    fig = render_prediction_scatter(preds)
    assert isinstance(fig, go.Figure)
