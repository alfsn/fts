"""Backtest figures (W3a): draw only. Every series comes computed from fts's
catalog (``RunCurves``, ``compare_curves``); returns are fractions and only
the axis format shows them as %."""

from typing import Mapping

import pandas as pd
import plotly.graph_objects as go
from quant_ui.plots.theme import BBG_DOWN, BBG_UP, setup_bloomberg_theme

setup_bloomberg_theme()

_CLOSE_COLOR = "#17E6A1"
_SIGNAL_COLORS = {"BUY": BBG_UP, "SELL": BBG_DOWN, "HOLD": "#FFA726"}


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    return fig


def plot_trade_overlay(close: pd.Series, fills: pd.DataFrame) -> go.Figure:
    """The traded close (NaN gaps stay gaps) with buy/sell fill markers.
    ``fills``: fill_timestamp, side, quantity, fill_price."""
    if close.dropna().empty and fills.empty:
        return _empty("No price or fill data")
    fig = go.Figure(
        go.Scatter(
            x=close.index,
            y=close,
            mode="lines",
            name="Close",
            line=dict(color=_CLOSE_COLOR, width=1.5),
        )
    )
    for side, symbol, color in (
        ("BUY", "triangle-up", BBG_UP),
        ("SELL", "triangle-down", BBG_DOWN),
    ):
        rows = fills[fills["side"].astype(str).str.upper() == side]
        if rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rows["fill_timestamp"],
                y=rows["fill_price"],
                mode="markers",
                name=f"{side.title()} fills",
                marker=dict(symbol=symbol, size=12, color=color),
                customdata=rows["quantity"],
                hovertemplate=f"<b>{side}</b><br>Price: %{{y:.4f}}"
                "<br>Quantity: %{customdata}<extra></extra>",
            )
        )
    fig.update_layout(title="Fills on Price", yaxis_title="Price (raw)")
    return fig


def plot_prediction_scatter(predictions: pd.DataFrame) -> go.Figure:
    """Confidence vs the realised future return, one colour per signal. A
    prediction without a realised return is not drawn (never put at 0)."""
    scored = predictions.dropna(subset=["confidence", "actual_future_return"])
    if scored.empty:
        return _empty("No scored predictions")
    fig = go.Figure()
    for signal, group in scored.groupby("predicted_signal"):
        fig.add_trace(
            go.Scatter(
                x=group["confidence"],
                y=group["actual_future_return"],
                mode="markers",
                name=f"{signal}",
                marker=dict(
                    size=8,
                    color=_SIGNAL_COLORS.get(str(signal).upper(), "#29B6F6"),
                    opacity=0.7,
                ),
            )
        )
    fig.update_layout(
        title=f"Confidence vs Realised Return (n={len(scored)} of {len(predictions)})",
        xaxis_title="Confidence",
        yaxis_title="Realised future return",
        yaxis_tickformat=".2%",
    )
    return fig


def plot_cumulative_comparison(
    curves: Mapping[str, pd.Series], x_title: str = "Date"
) -> go.Figure:
    """Several cumulative returns (fractions) on one chart."""
    if not curves:
        return _empty("No runs to compare")
    fig = go.Figure(
        [
            go.Scatter(x=curve.index, y=curve, mode="lines", name=label)
            for label, curve in curves.items()
        ]
    )
    fig.update_layout(
        title="Cumulative Return, each run from its own first bar",
        xaxis_title=x_title,
        yaxis_tickformat=".1%",
        hovermode="x unified",
    )
    return fig
