import numpy as np
import pandas as pd
import plotly.graph_objects as go
from quant_ui.plots.performance import plot_backtest_performance


def test_plot_backtest_performance():
    # Mock PnL dataframe
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    df_pnl = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.linspace(100, 150, 50),
            "high": np.linspace(105, 155, 50),
            "low": np.linspace(95, 145, 50),
            "close": np.linspace(102, 152, 50),
            "side": ["BUY", "SELL"] * 25,
            "size": [1.0] * 50,
            "price": np.linspace(102, 152, 50),
            "cum_strat_return": np.linspace(0, 0.5, 50),
            "cum_baseline_return": np.linspace(0, 0.4, 50),
            "actual_equity": np.linspace(10000, 15000, 50),
            "cash": np.linspace(5000, 5000, 50),
            "equity_position": np.linspace(1, 1, 50),
        }
    )

    fig = plot_backtest_performance(df_pnl, instrument_id="BTC/USD")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3  # Should have candlestick and strategy line at minimum

    # Verify the subplot title is set
    titles = [ann.text for ann in fig.layout.annotations if getattr(ann, "text", None)]
    assert any("BTC/USD" in t for t in titles)


def test_plot_backtest_performance_empty_trades():
    # Test without trade signals and actual equity
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    df_pnl = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100] * 10,
            "high": [105] * 10,
            "low": [95] * 10,
            "close": [102] * 10,
            "cum_strat_return": [0.1] * 10,
        }
    )

    fig = plot_backtest_performance(df_pnl, instrument_id="ETH/USD")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # Candlestick + strategy line
