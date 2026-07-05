# src/trading_bot/dashboard/renderers.py

"""
Plotly Visual Renderers for Quant Data Catalog & Backtest Explorer.
"""

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_bot.core.schemas import BacktestDetailDTO


def render_equity_curve(
    detail_dto: BacktestDetailDTO, normalize: bool = False
) -> go.Figure:
    """
    Renders an interactive Plotly figure with 2 subplots:
    - Top Subplot: Account Equity curve ($ or % return)
    - Bottom Subplot: Underwater Drawdown (% drop from peak)
    """
    if not detail_dto.equity_curve:
        fig = go.Figure()
        fig.add_annotation(text="No equity curve data available", showarrow=False)
        return fig

    df = pd.DataFrame(detail_dto.equity_curve)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    initial_eq = (
        df["equity"].iloc[0] if len(df) > 0 and df["equity"].iloc[0] > 0 else 1.0
    )

    if normalize:
        df["display_equity"] = ((df["equity"] - initial_eq) / initial_eq) * 100.0
        y_title = "Return (%)"
        hover_fmt = ".2f"
        suffix = "%"
    else:
        df["display_equity"] = df["equity"]
        y_title = "Equity ($)"
        hover_fmt = ",.2f"
        suffix = ""

    # Compute underwater drawdown series
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["peak"] - df["equity"]) / df["peak"] * 100.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            f"Backtest Equity Curve ({detail_dto.run_id})",
            "Underwater Drawdown (%)",
        ],
    )

    # Top plot: Equity
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["display_equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#00C805", width=2),
            hovertemplate=f"%{{y:{hover_fmt}}}{suffix}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Bottom plot: Drawdown
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=-df["drawdown"],
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#FF3B30", width=1.5),
            fillcolor="rgba(255, 59, 48, 0.2)",
            hovertemplate="%{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)

    return fig


def render_trade_overlay(equity_curve: List[Dict], trades: List[Dict]) -> go.Figure:
    """
    Renders price bar close line with filled trade execution markers:
    - Green Triangle-Up for BUY / LONG fills
    - Red Triangle-Down for SELL / SHORT fills
    """
    if not equity_curve:
        fig = go.Figure()
        fig.add_annotation(text="No price data available", showarrow=False)
        return fig

    df_eq = pd.DataFrame(equity_curve)
    df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"])
    df_eq = df_eq.sort_values("timestamp")

    fig = go.Figure()

    # Price close line
    fig.add_trace(
        go.Scatter(
            x=df_eq["timestamp"],
            y=df_eq["close"],
            mode="lines",
            name="Market Close",
            line=dict(color="#17E6A1", width=1.5),
        )
    )

    if trades:
        df_trades = pd.DataFrame(trades)
        if "fill_timestamp" in df_trades.columns:
            df_trades["fill_timestamp"] = pd.to_datetime(df_trades["fill_timestamp"])

            buys = df_trades[df_trades["side"].astype(str).str.upper() == "BUY"]
            sells = df_trades[df_trades["side"].astype(str).str.upper() == "SELL"]

            if not buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buys["fill_timestamp"],
                        y=buys["fill_price"],
                        mode="markers",
                        name="Buy Fills",
                        marker=dict(
                            symbol="triangle-up",
                            size=12,
                            color="#00E676",
                            line=dict(width=1, color="white"),
                        ),
                        text=buys["fill_size"].apply(lambda s: f"Size: {s}"),
                        hovertemplate="<b>BUY</b><br>Price: %{y:.4f}<br>%{text}<extra></extra>",
                    )
                )

            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells["fill_timestamp"],
                        y=sells["fill_price"],
                        mode="markers",
                        name="Sell Fills",
                        marker=dict(
                            symbol="triangle-down",
                            size=12,
                            color="#FF5252",
                            line=dict(width=1, color="white"),
                        ),
                        text=sells["fill_size"].apply(lambda s: f"Size: {s}"),
                        hovertemplate="<b>SELL</b><br>Price: %{y:.4f}<br>%{text}<extra></extra>",
                    )
                )

    fig.update_layout(
        template="plotly_dark",
        title="Trade Execution Overlay on Price",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_title="Timestamp",
        yaxis_title="Asset Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_backtest_comparison(
    details_list: List[BacktestDetailDTO], align_by_index: bool = False
) -> go.Figure:
    """
    Renders multi-run backtest equity curves overlayed together, normalized to % growth.
    """
    fig = go.Figure()

    colors = [
        "#00E676",
        "#29B6F6",
        "#AB47BC",
        "#FFA726",
        "#FF5252",
        "#26A69A",
        "#EC407A",
    ]

    for idx, detail in enumerate(details_list):
        if not detail.equity_curve:
            continue

        df = pd.DataFrame(detail.equity_curve)
        initial_eq = (
            df["equity"].iloc[0] if len(df) > 0 and df["equity"].iloc[0] > 0 else 1.0
        )
        df["pct_return"] = ((df["equity"] - initial_eq) / initial_eq) * 100.0

        color = colors[idx % len(colors)]
        label = f"{detail.run_id} ({detail.strategy_name})"

        if align_by_index:
            x_vals = list(range(len(df)))
            x_title = "Step Index (Ticks)"
        else:
            x_vals = pd.to_datetime(df["timestamp"])
            x_title = "Calendar Date"

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df["pct_return"],
                mode="lines",
                name=label,
                line=dict(width=2, color=color),
                hovertemplate=f"<b>{detail.run_id}</b><br>Return: %{{y:.2f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title="Side-by-Side Backtest Equity Comparison (% Cumulative Return)",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_title=x_title,
        yaxis_title="Return (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_prediction_scatter(predictions: List[Dict]) -> go.Figure:
    """
    Scatter plot displaying prediction confidence vs actual future return.
    """
    if not predictions:
        fig = go.Figure()
        fig.add_annotation(text="No prediction logs available", showarrow=False)
        return fig

    df = pd.DataFrame(predictions)
    df = df.dropna(subset=["confidence"])

    fig = go.Figure()

    # Color by signal
    signal_colors = {"BUY": "#00E676", "SELL": "#FF5252", "HOLD": "#FFA726"}

    for signal, group in df.groupby("predicted_signal"):
        color = signal_colors.get(str(signal).upper(), "#29B6F6")
        y_vals = (
            group["actual_future_return"]
            if "actual_future_return" in group.columns
            else [0.0] * len(group)
        )

        fig.add_trace(
            go.Scatter(
                x=group["confidence"],
                y=y_vals,
                mode="markers",
                name=f"Signal: {signal}",
                marker=dict(size=8, color=color, opacity=0.7),
                hovertemplate=f"<b>Signal: {signal}</b><br>Confidence: %{{x:.3f}}<br>Future Return: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title="ML Model Prediction Confidence Distribution",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_title="Prediction Confidence Score",
        yaxis_title="Actual Future Return",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
