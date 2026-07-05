# src/trading_bot/dashboard/app.py

"""
Streamlit Multi-Page Interactive Dashboard for Quant Data Catalog & Backtest Explorer.
"""

import os
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_bot.config import get_settings
from trading_bot.core.catalog_repository import CatalogQueryService
from trading_bot.core.schemas import BacktestDetailDTO, ModelDetailDTO
from trading_bot.dashboard.renderers import (
    render_backtest_comparison,
    render_equity_curve,
    render_prediction_scatter,
    render_trade_overlay,
)

# Page Configuration
st.set_page_config(
    page_title="Quant Data Catalog & Backtest Visibility",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Database Session Factory
db_url = os.getenv("DATABASE_URL", get_settings().DATABASE_URL)
engine = create_engine(
    db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
)
SessionFactory = sessionmaker(bind=engine)
query_service = CatalogQueryService(SessionFactory)


# Caching wrappers for service calls
@st.cache_data(ttl=30)
def get_cached_db_summary() -> Dict[str, int]:
    return query_service.get_database_summary()


@st.cache_data(ttl=60)
def get_cached_models(
    model_type: Optional[str] = None,
    market_id: Optional[str] = None,
    status: Optional[str] = None,
):
    return query_service.model_repo.list_models(
        model_type=model_type, market_id=market_id, status=status
    )


@st.cache_data(ttl=60)
def get_cached_model_details(model_id: str) -> Optional[ModelDetailDTO]:
    return query_service.model_repo.get_model_details(model_id)


@st.cache_data(ttl=60)
def get_cached_backtest_runs(
    market_id: Optional[str] = None, min_sharpe: Optional[float] = None
):
    return query_service.backtest_repo.list_runs(
        market_id=market_id, min_sharpe=min_sharpe
    )


@st.cache_data(ttl=60)
def get_cached_run_details(run_id: str) -> Optional[BacktestDetailDTO]:
    return query_service.backtest_repo.get_run_details(run_id)


# Main Layout header
st.title("⚡ Quant System Data Catalog & Model Registry")
st.caption(f"Connected Database: `{db_url}`")

# Sidebar Metrics
st.sidebar.header("System Health & Database Summary")
db_stats = get_cached_db_summary()

col_sb1, col_sb2 = st.sidebar.columns(2)
col_sb1.metric("Markets", db_stats.get("markets", 0))
col_sb2.metric("Bar Ticks", f"{db_stats.get('bar_logs', 0):,}")

col_sb3, col_sb4 = st.sidebar.columns(2)
col_sb3.metric("Total Models", db_stats.get("models", 0))
col_sb4.metric("Production", db_stats.get("production_models", 0))

st.sidebar.metric("Equity Log Ticks", f"{db_stats.get('equity_log_ticks', 0):,}")
st.sidebar.divider()

# Navigation Tabs
tab_overview, tab_models, tab_backtests = st.tabs(
    ["📊 Database Overview", "🤖 Model Registry Explorer", "📈 Backtest Run Explorer"]
)


# --- TAB 1: Database Overview ---
with tab_overview:
    st.subheader("Database Table Health & Persistence Summary")
    st.markdown(
        "Overview of persisted historical bar data, ML models, trade logs, and simulation equity ticks."
    )

    cols = st.columns(4)
    cols[0].metric("Market Entities", db_stats.get("markets", 0))
    cols[1].metric("OHLCV Bar Logs", f"{db_stats.get('bar_logs', 0):,}")
    cols[2].metric("Order Logs", f"{db_stats.get('order_logs', 0):,}")
    cols[3].metric("Trade Fill Logs", f"{db_stats.get('trade_logs', 0):,}")

    cols2 = st.columns(4)
    cols2[0].metric("Registered Models", db_stats.get("models", 0))
    cols2[1].metric("Production Models", db_stats.get("production_models", 0))
    cols2[2].metric("Backtest Equity Ticks", f"{db_stats.get('equity_log_ticks', 0):,}")
    cols2[3].metric("Prediction Logs", f"{db_stats.get('prediction_logs', 0):,}")

    st.divider()
    st.info(
        "💡 Tip: Navigate to the **Model Registry Explorer** to promote models to Production or inspect hyperparameters. Use **Backtest Run Explorer** to overlay equity curves."
    )


# --- TAB 2: Model Registry Explorer ---
with tab_models:
    st.subheader("Model Registry Explorer & Promotion Lifecycle")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        status_filter = st.selectbox(
            "Status Filter", ["ALL", "candidate", "production", "archived"], index=0
        )
    with col_f2:
        model_type_filter = st.text_input(
            "Filter by Model Type (e.g. LightGBM, XGBoost)"
        )
    with col_f3:
        market_filter = st.text_input("Filter by Market ID")

    st_filter = None if status_filter == "ALL" else status_filter
    mt_filter = model_type_filter.strip() if model_type_filter.strip() else None
    mk_filter = market_filter.strip() if market_filter.strip() else None

    models = get_cached_models(
        model_type=mt_filter, market_id=mk_filter, status=st_filter
    )

    if not models:
        st.warning("No models found matching the selected filters.")
    else:
        # Table view
        data_dicts = []
        for m in models:
            data_dicts.append(
                {
                    "Model ID": m.model_id,
                    "Type": m.model_type,
                    "Market": m.market_id,
                    "Interval": m.interval,
                    "Horizon": m.horizon,
                    "Status": m.status,
                    "ONNX Path": m.onnx_path,
                    "Created At": (
                        m.created_at.strftime("%Y-%m-%d %H:%M") if m.created_at else ""
                    ),
                }
            )
        df_models = pd.DataFrame(data_dicts)
        st.dataframe(df_models, use_container_width=True)

        st.divider()
        st.subheader("Model Deep-Dive & Promotion Controls")

        selected_model_id = st.selectbox(
            "Select Model to Inspect / Promote", [m.model_id for m in models]
        )
        if selected_model_id:
            detail = get_cached_model_details(selected_model_id)
            if detail:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status", detail.status.upper())
                c2.metric("Model Type", detail.model_type)
                c3.metric(
                    "Market / Horizon", f"{detail.market_id} (h={detail.horizon})"
                )
                c4.metric(
                    "ONNX Artifact",
                    "✅ Available" if detail.onnx_exists else "⚠️ Missing",
                )

                # Promotion Action
                if detail.status != "production":
                    if st.button(f"🚀 Promote '{detail.model_id}' to PRODUCTION"):
                        success = query_service.model_repo.update_model_status(
                            detail.model_id, "production"
                        )
                        if success:
                            st.success(
                                f"Successfully promoted model '{detail.model_id}' to PRODUCTION! Any prior production model for signature demoted to 'candidate'."
                            )
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error("Failed to update model status.")
                else:
                    if st.button(f"📦 Archive Model '{detail.model_id}'"):
                        query_service.model_repo.update_model_status(
                            detail.model_id, "archived"
                        )
                        st.cache_data.clear()
                        st.rerun()

                st.subheader("Hyperparameters & Evaluation Metrics")
                col_hp, col_met = st.columns(2)
                with col_hp:
                    st.write("**Hyperparameters**")
                    st.json(detail.hyperparameters)
                with col_met:
                    st.write("**Evaluation Metrics**")
                    st.json(detail.metrics)


# --- TAB 3: Backtest Run Explorer ---
with tab_backtests:
    st.subheader("Backtest Simulation Explorer & Performance Visualization")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        bk_market_filter = st.text_input("Filter Backtests by Market ID", value="")
    with col_b2:
        min_sharpe_filter = st.number_input(
            "Minimum Sharpe Ratio Filter", value=-10.0, step=0.5
        )

    bk_m_filter = bk_market_filter.strip() if bk_market_filter.strip() else None
    runs = get_cached_backtest_runs(market_id=bk_m_filter, min_sharpe=min_sharpe_filter)

    if not runs:
        st.warning("No backtest runs found matching the filter criteria.")
    else:
        # Run Summary Table
        run_table_data = []
        for r in runs:
            run_table_data.append(
                {
                    "Run ID": r.run_id,
                    "Strategy": r.strategy_name,
                    "Market": r.market_id,
                    "Total Return (%)": r.total_return,
                    "Sharpe Ratio": r.sharpe_ratio,
                    "Max Drawdown (%)": r.max_drawdown,
                    "Win Rate (%)": r.win_rate,
                    "Total Trades": r.total_trades,
                    "Start Time": (
                        r.start_time.strftime("%Y-%m-%d %H:%M") if r.start_time else ""
                    ),
                    "End Time": (
                        r.end_time.strftime("%Y-%m-%d %H:%M") if r.end_time else ""
                    ),
                }
            )
        df_runs = pd.DataFrame(run_table_data)
        st.dataframe(df_runs, use_container_width=True)

        st.divider()

        # Detailed Run Visualizer
        st.subheader("Single Run Deep-Dive Visualizer")
        selected_run_id = st.selectbox(
            "Select Run ID for Time-Series Analysis", [r.run_id for r in runs]
        )

        if selected_run_id:
            run_detail = get_cached_run_details(selected_run_id)
            if run_detail:
                # Key performance metric cards
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Return", f"{run_detail.total_return:+.2f}%")
                m2.metric("Sharpe Ratio", f"{run_detail.sharpe_ratio:.2f}")
                m3.metric(
                    "Max Drawdown",
                    f"{run_detail.max_drawdown:.2f}%",
                    delta_color="inverse",
                )
                m4.metric("Win Rate", f"{run_detail.win_rate:.1f}%")
                m5.metric("Total Trades", run_detail.total_trades)

                norm_toggle = st.checkbox(
                    "Normalize Equity Curve to Percentage Return (%)", value=True
                )
                fig_equity = render_equity_curve(run_detail, normalize=norm_toggle)
                st.plotly_chart(fig_equity, use_container_width=True)

                col_overlay, col_pred = st.columns(2)
                with col_overlay:
                    fig_trades = render_trade_overlay(
                        run_detail.equity_curve, run_detail.trades
                    )
                    st.plotly_chart(fig_trades, use_container_width=True)
                with col_pred:
                    fig_pred = render_prediction_scatter(run_detail.predictions)
                    st.plotly_chart(fig_pred, use_container_width=True)

        st.divider()

        # Multi-Run Side-by-Side Comparison Tool
        st.subheader("Multi-Run Side-by-Side Comparison Tool")
        multi_selected_runs = st.multiselect(
            "Select Backtest Runs to Compare",
            [r.run_id for r in runs],
            default=[r.run_id for r in runs[: min(3, len(runs))]],
        )

        if multi_selected_runs:
            compare_details = [
                get_cached_run_details(r_id) for r_id in multi_selected_runs
            ]
            compare_details = [d for d in compare_details if d is not None]

            if compare_details:
                align_mode = st.radio(
                    "Comparison Timeline Alignment",
                    ["Calendar Timestamp", "T=0 Step Offset (Index Aligned)"],
                    horizontal=True,
                )
                align_by_idx = align_mode != "Calendar Timestamp"

                fig_comp = render_backtest_comparison(
                    compare_details, align_by_index=align_by_idx
                )
                st.plotly_chart(fig_comp, use_container_width=True)
