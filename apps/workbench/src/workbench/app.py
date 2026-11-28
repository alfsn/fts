"""The workbench shell: navigation and the database sidebar (W3a).

Run with `uv run workbench` (localhost only)."""

from pathlib import Path

import streamlit as st
from workbench import settings, state

VIEWS = Path(__file__).with_name("views")

st.set_page_config(page_title="Workbench", page_icon="📈", layout="wide")

with st.sidebar:
    st.header("Databases")
    st.text_input(
        "fts research DB (models, runs)",
        value=settings.research_url(),
        key=state.RESEARCH_KEY,
        help="SQLAlchemy URL. Blank: fts's DATABASE_URL default.",
    )
    st.text_input(
        "Price cache (shared)",
        value=settings.market_cache_url(),
        key=state.CACHE_KEY,
        help="SQLAlchemy URL. Blank: MARKET_CACHE_URL, else the workspace's market_cache.db.",
    )
    summary = state.catalog().get_database_summary()
    a, b = st.columns(2)
    a.metric("Cached bars", f"{summary['bar_logs']:,}")
    b.metric("Markets", summary["markets"])
    a.metric("Models", summary["models"])
    b.metric("Production", summary["production_models"])
    st.metric("Equity rows", f"{summary['equity_log_ticks']:,}")

page = st.navigation(
    [
        st.Page(VIEWS / "portfolio.py", title="Portfolio", icon="💼", default=True),
        st.Page(VIEWS / "models.py", title="Models", icon="🤖"),
        st.Page(VIEWS / "backtests.py", title="Backtests", icon="📈"),
        st.Page(VIEWS / "cache.py", title="Cache", icon="🗄️"),
        st.Page(VIEWS / "fx.py", title="FX", icon="💱"),
    ]
)
page.run()
