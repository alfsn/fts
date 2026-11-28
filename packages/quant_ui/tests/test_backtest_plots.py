"""Backtest figures, moved from fts's dashboard renderers (W3a)."""

import pandas as pd
from quant_ui.plots.backtest import (
    plot_cumulative_comparison,
    plot_prediction_scatter,
    plot_trade_overlay,
)

INDEX = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")


def test_trade_overlay_draws_buys_and_sells_by_quantity():
    close = pd.Series([10.0, float("nan"), 12.0], index=INDEX)
    fills = pd.DataFrame(
        {
            "fill_timestamp": INDEX[[0, 2]],
            "side": ["BUY", "SELL"],
            "quantity": [1.0, 1.0],
            "fill_price": [10.0, 12.0],
        }
    )
    fig = plot_trade_overlay(close, fills)
    assert [t.name for t in fig.data] == ["Close", "Buy fills", "Sell fills"]


def test_prediction_without_a_realised_return_is_not_drawn_at_zero():
    predictions = pd.DataFrame(
        {
            "predicted_signal": ["BUY", "SELL", "BUY"],
            "confidence": [0.8, 0.7, 0.6],
            "actual_future_return": [0.02, -0.01, float("nan")],
        }
    )
    fig = plot_prediction_scatter(predictions)
    ys = [y for trace in fig.data for y in trace.y]
    assert sorted(ys) == [-0.01, 0.02]


def test_comparison_draws_one_line_per_run():
    curves = {
        "a": pd.Series([0.0, 0.1], index=[0.0, 1.0]),
        "b": pd.Series([0.0, 0.05], index=[0.0, 0.5]),
    }
    fig = plot_cumulative_comparison(curves, x_title="Days since start")
    assert [t.name for t in fig.data] == ["a", "b"]
