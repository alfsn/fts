"""Market-data figures (W3b): the cache's raw and adjusted prices, and the
ARS CCL. Draw only: every series comes computed from ``quant_data.inspect``;
missing values stay gaps."""

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go
from quant_core.enums import CorporateActionType
from quant_core.models import CorporateAction
from quant_ui.plots.theme import BBG_DOWN, BBG_TEXT, setup_bloomberg_theme

setup_bloomberg_theme()

_RAW = "#78909c"
_SPLIT = "#2196f3"
_TOTAL = "#ffb300"


def plot_price_adjustments(
    raw: pd.Series,
    split: pd.Series,
    total_return: pd.Series,
    actions: Sequence[CorporateAction] = (),
    title: str = "Price",
) -> go.Figure:
    """Raw closes beside split- and total-return-adjusted ones, with a
    dashed line on every split and a dotted one on every dividend ex-date."""
    fig = go.Figure(
        [
            go.Scatter(
                x=raw.index, y=raw, name="Raw (as traded)", line=dict(color=_RAW)
            ),
            go.Scatter(
                x=split.index, y=split, name="Split-adjusted", line=dict(color=_SPLIT)
            ),
            go.Scatter(
                x=total_return.index,
                y=total_return,
                name="Total return (dividends reinvested)",
                line=dict(color=_TOTAL, dash="dot"),
            ),
        ]
    )
    for action in actions:
        is_split = action.action_type is CorporateActionType.SPLIT
        fig.add_vline(
            x=action.ex_date,
            line_dash="dash" if is_split else "dot",
            line_color=BBG_DOWN if is_split else BBG_TEXT,
            opacity=0.6,
        )
    fig.update_layout(title=title, yaxis_title="Price", hovermode="x unified")
    return fig


def plot_ccl(ccl: pd.Series, implied: pd.DataFrame) -> go.Figure:
    """The CCL (ARS per USD) and the rate each pair implies alone. Days where
    the pairs disagree are holes in the CCL line."""
    fig = go.Figure(
        [
            go.Scatter(
                x=implied.index, y=implied[c], name=c, line=dict(width=1), opacity=0.6
            )
            for c in implied.columns
        ]
    )
    fig.add_trace(
        go.Scatter(x=ccl.index, y=ccl, name="CCL", line=dict(color=_TOTAL, width=2))
    )
    fig.update_layout(
        title="CCL (ARS per USD)", yaxis_title="ARS per USD", hovermode="x unified"
    )
    return fig
