"""FX: the ARS CCL from the cached GGAL and YPF pairs, the same recipe the
analyser values ARS with, beside each pair's own rate (W3b). Read-only."""

import streamlit as st
from quant_data import inspect
from quant_ui.plots.market import plot_ccl
from workbench import state


@st.cache_data(ttl=60)
def _view(cache):
    return inspect.ccl_view(cache or None)


st.title("ARS CCL")
view = _view(state.market_cache_url())
if view.missing:
    st.warning(
        "No cached daily bars for " + ", ".join(view.missing) + ". A Portfolio "
        "review of an ARS ledger caches them."
    )
if view.implied.empty:
    st.stop()

st.plotly_chart(plot_ccl(view.ccl, view.implied), use_container_width=True)
a, b, c = st.columns(3)
a.metric("Days with a CCL", int(view.ccl.notna().sum()))
b.metric("Rejected (pairs > 5% apart)", view.rejected_days)
c.metric("One pair only (unchecked)", view.single_pair_days)
st.caption(
    "Each pair's rate = local price × shares per ADR / ADR price, on "
    "split-adjusted closes (the ratios are today's). The analyser stores the "
    "inverse, USD per ARS. Buenos Aires closes before New York, so daily "
    "rates carry timing noise."
)
