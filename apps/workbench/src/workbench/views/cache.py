"""Cache: what the shared price cache holds, and one series raw beside its
adjusted views, with its corporate actions and gaps. Read-only: this page
never calls a vendor (W3b)."""

import pandas as pd
import streamlit as st
from quant_data import inspect
from quant_ui.plots.market import plot_price_adjustments
from workbench import state


@st.cache_data(ttl=60)
def _series(cache):
    return inspect.cached_series(cache or None)


@st.cache_data(ttl=60)
def _view(cache, symbol, interval):
    return inspect.inspect_series(symbol, interval, cache or None)


st.title("Price cache")
cache = state.market_cache_url()
rows = _series(cache)
if not rows:
    st.info(
        "The cache is empty. Fill it with `ingest_historical` (fts) or by running "
        "a Portfolio review; this page only reads it."
    )
    st.stop()

st.dataframe(pd.DataFrame([r.model_dump() for r in rows]), use_container_width=True)
labels = {f"{r.instrument_id} · {r.interval}": r for r in rows}
picked = labels[st.selectbox("Series", list(labels))]
try:
    view = _view(cache, picked.instrument_id, picked.interval)
except ValueError as e:  # a dividend not below the previous close
    st.error(str(e))
    st.stop()

st.plotly_chart(
    plot_price_adjustments(
        view.raw, view.split, view.total_return, view.actions, view.instrument_id
    ),
    use_container_width=True,
)
st.caption(
    "Raw is what traded (what valuation uses). Split-adjusted removes split "
    "jumps; total return also reinvests dividends, which lowers past prices, "
    "so only its latest price equals raw."
)

left, right = st.columns(2)
left.subheader("Corporate actions")
if view.actions:
    left.dataframe(
        pd.DataFrame(
            [
                {
                    "Ex-date": a.ex_date,
                    "Type": a.action_type.value,
                    "Ratio": a.ratio,
                    "Amount": a.amount,
                }
                for a in view.actions
            ]
        )
    )
else:
    left.caption("None cached.")
right.subheader("Gaps")
rule = (
    f"{view.source}'s rule" if view.known_source else "the default exchange-hours rule"
)
right.caption(f"A gap is more than {view.max_gap} between bars ({rule}).")
if view.gaps.empty:
    right.caption("No gaps.")
else:
    right.dataframe(view.gaps)
