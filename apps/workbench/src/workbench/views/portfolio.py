"""Portfolio: the analyser's review of a YAML ledger, the same pipeline and
charts as `portfolio-analyser client-review` (W2), over a date window."""

from dataclasses import asdict

import pandas as pd
import streamlit as st
from portfolio_analyser import review
from portfolio_analyser.visualization.plot_factory import backward_analytics_figures
from quant_core.models import MissingFXRateError
from workbench import state

ARGS_KEY = "portfolio_args"


@st.cache_data(show_spinner="Valuing the ledger (fills the price cache on first run)…")
def _review(path, cache, benchmark, risk_free, withholding, start, end):
    result = review.prepare_review(
        review.load_portfolio(path),
        cache or None,
        benchmark_symbol=benchmark or None,
        risk_free_rate=risk_free,
        withholding=withholding,
        start=start,
        end=end,
    )
    figures = backward_analytics_figures(
        result.values,
        result.flows,
        benchmark=result.benchmark,
        benchmark_name=result.benchmark_name,
        risk_free_rate=result.risk_free,
    )
    return result, figures


st.title("Portfolio review")
st.caption("YAML broker only: the IBKR / Inviu / IOL connectors are mocks.")

with st.form("review"):
    path = st.text_input("Portfolio YAML (with a ledger)", str(review.DEMO_LEDGER))
    a, b, c = st.columns(3)
    benchmark = a.text_input("Benchmark (blank: the YAML's, else SPY)")
    rf_text = b.text_input(
        "Risk-free rate, annual fraction (blank: daily ^IRX, USD only)"
    )
    withholding = c.number_input(
        "Dividend withholding on the benchmark", 0.0, 0.99, 0.0, 0.05
    )
    d, e = st.columns(2)
    start = d.date_input("Start (blank: first ledger day)", value=None)
    end = e.date_input("End (blank: valuation date)", value=None)
    submitted = st.form_submit_button("Run")

if submitted:
    try:
        risk_free = float(rf_text) if rf_text.strip() else None
    except ValueError:
        st.error(f"Risk-free rate must be a fraction such as 0.04, not {rf_text!r}.")
        st.stop()
    st.session_state[ARGS_KEY] = (
        path,
        state.market_cache_url(),
        benchmark.strip(),
        risk_free,
        withholding,
        pd.Timestamp(start) if start else None,
        pd.Timestamp(end) if end else None,
    )

args = st.session_state.get(ARGS_KEY)
if args is None:
    st.info("Pick a ledger and press **Run**. The demo ledger is filled in.")
    st.stop()

try:
    result, figures = _review(*args)
except review.ReviewError as e:
    st.error(str(e))
    if e.breaks is not None:
        st.code(e.breaks.describe())
    st.stop()
except (MissingFXRateError, FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

snapshot = result.snapshot
first, last = result.period
base = snapshot.portfolio.base_currency.value
m1, m2, m3 = st.columns(3)
m1.metric(f"Total value ({base})", f"{snapshot.total_value:,.2f}")
m2.metric("Valued at close", str(snapshot.as_of.date()))
if snapshot.ars_per_usd is not None:
    m3.metric("CCL (ARS per USD)", f"{snapshot.ars_per_usd:,.2f}")
st.caption(
    f"Window {first.date()} → {last.date()}: {len(result.values)} valued days, "
    f"{len(result.dropped)} dropped (unpriced). KPIs rebase on the last close "
    "before the start."
)
for warning in result.warnings:
    st.warning(warning)
if snapshot.breaks is not None and snapshot.breaks.describe():
    st.code(snapshot.breaks.describe())

left, right = st.columns(2)
left.subheader("Cash")
left.dataframe(pd.DataFrame([asdict(r) for r in review.cash_rows(snapshot)]))
right.subheader("Positions")
right.dataframe(pd.DataFrame([asdict(r) for r in review.position_rows(snapshot)]))

for figure in figures:
    st.plotly_chart(figure, use_container_width=True)
