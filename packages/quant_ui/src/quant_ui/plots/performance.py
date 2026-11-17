import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quant_ui.plots.theme import BBG_DOWN, BBG_UP, setup_bloomberg_theme

setup_bloomberg_theme()


def plot_backtest_performance(df_pnl: pd.DataFrame, instrument_id: str) -> go.Figure:
    """
    Pure UI function: Takes a Pandas dataframe with PnL and trade signals,
    returns a Plotly Figure containing the candlestick and cumulative returns.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{instrument_id} Price Action & Executed Trades",
            "Performance: Cumulative Strategy Returns vs Buy & Hold",
        ),
        row_heights=[0.6, 0.4],
    )

    # Subplot 1: Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df_pnl["timestamp"],
            open=df_pnl["open"],
            high=df_pnl["high"],
            low=df_pnl["low"],
            close=df_pnl["close"],
            name="OHLC",
            increasing_line_color=BBG_UP,
            decreasing_line_color=BBG_DOWN,
        ),
        row=1,
        col=1,
    )

    # Overlays: Trades
    buys = (
        df_pnl[df_pnl["side"].astype(str).str.lower() == "buy"]
        if "side" in df_pnl.columns
        else pd.DataFrame()
    )
    sells = (
        df_pnl[df_pnl["side"].astype(str).str.lower() == "sell"]
        if "side" in df_pnl.columns
        else pd.DataFrame()
    )

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["timestamp"],
                y=buys["close"],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color=BBG_UP,
                    line=dict(width=1, color="black"),
                ),
                name="BUY Fill",
                hovertext=buys.apply(
                    lambda r: f"BUY {r.get('size', 1.0):.2f} @ {r.get('price', r['close']):.2f}",
                    axis=1,
                ),
            ),
            row=1,
            col=1,
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["timestamp"],
                y=sells["close"],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=BBG_DOWN,
                    line=dict(width=1, color="black"),
                ),
                name="SELL Fill",
                hovertext=sells.apply(
                    lambda r: f"SELL {r.get('size', 1.0):.2f} @ {r.get('price', r['close']):.2f}",
                    axis=1,
                ),
            ),
            row=1,
            col=1,
        )

    # Subplot 2: Cumulative P&L
    if "actual_equity" in df_pnl.columns:
        hovertext = df_pnl.apply(
            lambda r: (
                f"Date: {r['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>"
                f"Return: {r['cum_strat_return']*100:.2f}%<br>"
                f"Equity: ${r['actual_equity']:.2f}<br>"
                f"Cash: ${r.get('cash', 0.0):.2f}<br>"
                f"Position: {r.get('equity_position', 0.0):.4f}"
            ),
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_pnl["timestamp"],
                y=df_pnl["cum_strat_return"] * 100,
                mode="lines",
                name="ML Strategy P&L",
                line=dict(color="#2196f3", width=2),
                hovertext=hovertext,
                hoverinfo="text",
            ),
            row=2,
            col=1,
        )
    else:
        if "cum_strat_return" in df_pnl.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_pnl["timestamp"],
                    y=df_pnl["cum_strat_return"] * 100,
                    mode="lines",
                    name="ML Strategy P&L",
                    line=dict(color="#2196f3", width=2),
                ),
                row=2,
                col=1,
            )

    if "cum_baseline_return" in df_pnl.columns:
        fig.add_trace(
            go.Scatter(
                x=df_pnl["timestamp"],
                y=df_pnl["cum_baseline_return"] * 100,
                mode="lines",
                name="Buy & Hold Baseline",
                line=dict(color="#78909c", width=1.5, dash="dash"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
