import json
import logging
import os
from typing import Dict, List, Optional

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output, display
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trading_bot.core.enums import OrderSide, SignalType

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """
    Provides interactive widgets and Plotly charts for Jupyter Notebooks
    to inspect model outputs, backtest P&L, risk/sizing decisions,
    and neural network filters/weights.
    """

    def __init__(self, db_url: str = "sqlite:///./dev.db") -> None:
        """
        Initializes the visualizer with a SQLite database URL.
        """
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False
        )

    def load_data(
        self, market_id: str, strategy_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Loads and joins bar logs, model predictions, and trade logs from the database
        for a given market and strategy.
        """
        session = self.SessionLocal()
        try:
            # 1. Query bar logs
            bars_query = f"""
                SELECT timestamp, open, high, low, close, volume, dollar_volume
                FROM bar_data_logs
                WHERE market_id = '{market_id}'
                ORDER BY timestamp ASC
            """
            df_bars = pd.read_sql(bars_query, session.bind)
            if df_bars.empty:
                logger.warning(f"No bar data found for market: {market_id}")
                return pd.DataFrame()

            df_bars["timestamp"] = pd.to_datetime(df_bars["timestamp"])

            # 2. Query model predictions
            pred_filter = ""
            if strategy_name:
                pred_filter = f"AND strategy_name = '{strategy_name}'"
            preds_query = f"""
                SELECT timestamp, strategy_name, prediction_output, predicted_signal, confidence, actual_future_return
                FROM model_prediction_logs
                WHERE market_id = '{market_id}' {pred_filter}
                ORDER BY timestamp ASC
            """
            df_preds = pd.read_sql(preds_query, session.bind)
            if not df_preds.empty:
                df_preds["timestamp"] = pd.to_datetime(df_preds["timestamp"])
                # Merge predictions with bar data
                df = pd.merge(df_bars, df_preds, on="timestamp", how="left")
            else:
                df = df_bars.copy()
                df["strategy_name"] = None
                df["prediction_output"] = None
                df["predicted_signal"] = None
                df["confidence"] = None
                df["actual_future_return"] = None

            # 3. Query trade logs
            trades_query = f"""
                SELECT fill_timestamp as timestamp, side, fill_size as size, fill_price as price
                FROM trade_logs
                WHERE market_id = '{market_id}'
                ORDER BY timestamp ASC
            """
            df_trades = pd.read_sql(trades_query, session.bind)
            if not df_trades.empty:
                df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
                # We will match trades to the nearest bar timestamp for overlaying
                df_trades = df_trades.sort_values("timestamp")
                df = df.sort_values("timestamp")
                df = pd.merge_asof(
                    df,
                    df_trades,
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta("1h"),
                )
            else:
                df["side"] = None
                df["size"] = None
                df["price"] = None

            # Sort chronologically
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        finally:
            session.close()

    def get_available_markets_and_strategies(self) -> Dict[str, List[str]]:
        """
        Retrieves all markets and strategies that have logged predictions in the DB.
        """
        session = self.SessionLocal()
        try:
            query = """
                SELECT DISTINCT market_id, strategy_name
                FROM model_prediction_logs
            """
            df = pd.read_sql(query, session.bind)
            if df.empty:
                # Fallback to general markets
                df_m = pd.read_sql(
                    "SELECT DISTINCT market_id FROM markets", session.bind
                )
                markets = df_m["market_id"].tolist() if not df_m.empty else []
                return {m: ["None"] for m in markets}

            result = {}
            for m in df["market_id"].unique():
                result[m] = df[df["market_id"] == m]["strategy_name"].tolist()
            return result
        except Exception:
            return {}
        finally:
            session.close()

    def calculate_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates cumulative strategy returns vs buy-and-hold baseline.
        """
        df = df.copy()
        df["raw_return"] = df["close"].pct_change().fillna(0.0)

        # Simple backtest simulation from signals:
        # BUY signal -> long position (+1)
        # SELL signal -> short position (-1)
        # FLAT/HOLD signal -> flat position (0)
        df["position"] = 0.0
        if "predicted_signal" in df.columns:
            df["position"] = (
                df["predicted_signal"]
                .str.lower()
                .map(
                    {
                        SignalType.BUY.value: 1.0,
                        SignalType.SELL.value: -1.0,
                        SignalType.FLAT.value: 0.0,
                        SignalType.HOLD.value: 0.0,
                    }
                )
                .ffill()
                .fillna(0.0)
            )

        # Shift position by 1 step to avoid lookahead bias (trade occurs at next bar open)
        df["strat_position"] = df["position"].shift(1).fillna(0.0)
        df["strat_return"] = df["strat_position"] * df["raw_return"]

        # Cumulative P&L
        df["cum_baseline_return"] = (1.0 + df["raw_return"]).cumprod() - 1.0
        df["cum_strat_return"] = (1.0 + df["strat_return"]).cumprod() - 1.0
        return df

    def render_charts(self, df: pd.DataFrame, market_id: str) -> go.Figure:
        """
        Generates a Plotly subplot showing:
        1. Candlestick chart with trade overlays.
        2. Cumulative returns (Strategy vs Baseline).
        """
        df_pnl = self.calculate_pnl(df)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                f"{market_id} Price Action & Executed Trades",
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
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1,
            col=1,
        )

        # Overlays: Trades
        buys = (
            df_pnl[df_pnl["side"].astype(str).str.lower() == OrderSide.BUY.value]
            if "side" in df_pnl.columns
            else pd.DataFrame()
        )
        sells = (
            df_pnl[df_pnl["side"].astype(str).str.lower() == OrderSide.SELL.value]
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
                        color="#4caf50",
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
                        color="#f44336",
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
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=700,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.update_yaxes(title_text="Price / Quote Currency", row=1, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=2, col=1)

        return fig

    def inspect_step(self, row: pd.Series) -> None:
        """
        Renders HTML table outputs for inspecting the model predictions (outputs),
        and risk manager sizing details for a specific step.
        """
        timestamp = row["timestamp"]
        close_price = row["close"]

        # Parse predictions
        predictions_str = row.get("prediction_output")
        signal = row.get("predicted_signal", "None")
        confidence = row.get("confidence", 0.0)

        predictions_data = None
        if predictions_str:
            try:
                predictions_data = json.loads(predictions_str)
                # Unwrap 2D batch dimension if present (e.g. [[value]] or [[p1, p2, p3]])
                if (
                    isinstance(predictions_data, list)
                    and len(predictions_data) == 1
                    and isinstance(predictions_data[0], list)
                ):
                    predictions_data = predictions_data[0]
            except Exception:
                pass

        # Display metadata
        display(widgets.HTML(f"<h3>Inspection for Tick: <b>{timestamp}</b></h3>"))

        # Create panels for side-by-side layout
        col_predictions = widgets.Output()
        col_sizing = widgets.Output()

        # Render predictions (model outputs)
        with col_predictions:
            display(widgets.HTML("<h4>📤 Model Output Predictions</h4>"))
            if predictions_data is not None:
                display(
                    widgets.HTML(
                        f"<p><b>Raw Model Output:</b> <code>{predictions_data}</code></p>"
                    )
                )

                # If the prediction is a probability vector [down, flat, up]
                if isinstance(predictions_data, list) and len(predictions_data) == 3:
                    labels = ["DOWN (baja)", "FLAT", "UP (sube)"]
                    probs_df = pd.DataFrame(
                        {"Class": labels, "Probability": predictions_data}
                    )

                    # Draw a mini bar chart
                    fig = go.Figure(
                        go.Bar(
                            x=probs_df["Class"],
                            y=probs_df["Probability"],
                            marker_color=["#ef5350", "#78909c", "#4caf50"],
                        )
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        height=200,
                        margin=dict(l=10, r=10, t=10, b=10),
                        yaxis=dict(range=[0, 1]),
                    )
                    display(fig)
                else:
                    # Single regression prediction
                    pred_val = (
                        predictions_data[0]
                        if isinstance(predictions_data, list)
                        else predictions_data
                    )
                    display(
                        widgets.HTML(
                            f"<p><b>Forecasted Return:</b> <code>{pred_val:.6f}</code></p>"
                        )
                    )
            else:
                display(
                    widgets.HTML(
                        "<p style='color:#b0bec5;'>No prediction output logged.</p>"
                    )
                )

        # Render Sizing and Kelly Inspector
        with col_sizing:
            display(widgets.HTML("<h4>⚖️ Risk & Kelly Sizing Details</h4>"))
            display(
                widgets.HTML(
                    f"<table style='width:100%; border-collapse:collapse;'>"
                    f"<tr><td style='padding:5px; border-bottom:1px solid #37474f;'><b>Signal Direction:</b></td><td style='padding:5px; border-bottom:1px solid #37474f;'>{signal}</td></tr>"
                    f"<tr><td style='padding:5px; border-bottom:1px solid #37474f;'><b>Confidence (p):</b></td><td style='padding:5px; border-bottom:1px solid #37474f;'>{confidence:.4f}</td></tr>"
                    f"<tr><td style='padding:5px; border-bottom:1px solid #37474f;'><b>Reference Price:</b></td><td style='padding:5px; border-bottom:1px solid #37474f;'>${close_price:.4f}</td></tr>"
                    f"</table>"
                )
            )

            if (
                signal.lower() in [SignalType.BUY.value, SignalType.SELL.value]
                and confidence > 0.5
            ):
                p = confidence
                q = 1.0 - p
                if close_price > 0 and close_price < 1.0:
                    # Prediction Market Contract Odds
                    price = close_price
                    if signal.lower() == SignalType.BUY.value:
                        b_odds = (1.0 - price) / price
                        f_kelly = (p * b_odds - q) / b_odds
                    else:
                        b_odds = price / (1.0 - price)
                        f_kelly = (q * b_odds - p) / b_odds
                else:
                    # Standard asset Kelly (simplistic model: 1:1 odds)
                    b_odds = 1.0
                    f_kelly = p - q

                f_safe = f_kelly * 0.5  # half kelly default

                display(
                    widgets.HTML(
                        f"<h5 style='margin-top:10px;'>Kelly Derivation</h5>"
                        f"<ul style='padding-left:20px; font-size:0.9em;'>"
                        f"<li><b>Losing Prob (q):</b> {q:.4f}</li>"
                        f"<li><b>Implied Odds (b):</b> {b_odds:.4f}</li>"
                        f"<li><b>Theoretical Kelly Fraction (f*):</b> {f_kelly*100:.2f}%</li>"
                        f"<li><b>Safe Fraction (0.5x f*):</b> {f_safe*100:.2f}%</li>"
                        f"</ul>"
                    )
                )

        # Render layout grid
        grid = widgets.GridspecLayout(1, 2, height="320px")
        grid[0, 0] = col_predictions
        grid[0, 1] = col_sizing
        display(grid)

    def plot_onnx_weights(self, onnx_path: str) -> None:
        """
        Loads an ONNX model file and plots heatmaps of its convolutional or dense weights.
        """
        if not os.path.exists(onnx_path):
            display(
                widgets.HTML(
                    f"<p style='color:#ef5350;'>ONNX model file not found: <b>{onnx_path}</b></p>"
                )
            )
            return

        try:
            import onnx
            from onnx import numpy_helper

            model = onnx.load(onnx_path)
            weights = {}

            # Extract initializers (weights)
            for initializer in model.graph.initializer:
                weights[initializer.name] = numpy_helper.to_array(initializer)

            if not weights:
                display(
                    widgets.HTML(
                        "<p>No weight initializers found in the ONNX graph.</p>"
                    )
                )
                return

            # Filter for weights of interest (e.g. weights with dim >= 2)
            options = [name for name, arr in weights.items() if len(arr.shape) >= 2]
            if not options:
                display(
                    widgets.HTML("<p>No multi-dimensional weight matrices found.</p>")
                )
                return

            dropdown = widgets.Dropdown(
                options=options,
                description="Select Weight Layer:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="400px"),
            )
            output_plot = widgets.Output()

            def on_layer_change(change):
                with output_plot:
                    clear_output(wait=True)
                    weight_name = change["new"]
                    weight_arr = weights[weight_name]

                    if len(weight_arr.shape) == 3:
                        # 3D: [out, in, kernel]
                        display(
                            widgets.HTML(
                                f"<h5>Shape: {weight_arr.shape} (Out Channels, In Channels, Kernel Size)</h5>"
                            )
                        )
                        weight_slice = weight_arr[0, :, :]
                        title = f"{weight_name} (Slice: Out Channel 0)"
                    elif len(weight_arr.shape) == 4:
                        # 4D: [out, in, height, width]
                        display(widgets.HTML(f"<h5>Shape: {weight_arr.shape}</h5>"))
                        weight_slice = weight_arr[0, 0, :, :]
                        title = f"{weight_name} (Slice: Out 0, In 0)"
                    else:
                        display(
                            widgets.HTML(
                                f"<h5>Shape: {weight_arr.shape} (Outputs, Inputs)</h5>"
                            )
                        )
                        weight_slice = weight_arr
                        title = weight_name

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=weight_slice,
                            colorscale="RdBu",
                            zmid=0.0,
                        )
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Heatmap of {title}",
                        height=350,
                        width=600,
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    display(fig)

            dropdown.observe(on_layer_change, names="value")
            display(dropdown)
            display(output_plot)

            # Fire initial change
            on_layer_change({"new": options[0]})

        except Exception as e:
            display(
                widgets.HTML(
                    f"<p style='color:#ef5350;'>Error loading ONNX model: {e}</p>"
                )
            )

    def show_dashboard(self, onnx_model_path: Optional[str] = None) -> None:
        """
        Builds the entire interactive dashboard layout and outputs it to the cell.
        """
        # 1. Get available selections
        meta_dict = self.get_available_markets_and_strategies()
        if not meta_dict:
            display(
                widgets.HTML(
                    "<p style='color:#ef5350;'><b>No backtest prediction logs found in database.</b><br/>"
                    "Make sure your simulator runs the pipeline with a DatabasePredictionLogger registered, "
                    "or check your configured database file.</p>"
                )
            )
            return

        markets = list(meta_dict.keys())

        # 2. Controls
        market_dd = widgets.Dropdown(
            options=markets,
            description="Market:",
            layout=widgets.Layout(width="200px"),
        )
        strategy_dd = widgets.Dropdown(
            options=meta_dict[markets[0]],
            description="Strategy:",
            layout=widgets.Layout(width="250px"),
        )

        load_btn = widgets.Button(
            description="Load Simulation Data",
            button_style="primary",
            layout=widgets.Layout(width="180px"),
        )

        control_box = widgets.HBox([market_dd, strategy_dd, load_btn])
        display(control_box)

        def on_market_change(change):
            m = change["new"]
            strategy_dd.options = meta_dict.get(m, ["None"])

        market_dd.observe(on_market_change, names="value")

        # Layout panels
        plot_panel = widgets.Output()
        inspector_panel = widgets.Output()
        onnx_panel = widgets.Output()

        display(plot_panel)
        display(inspector_panel)
        if onnx_model_path:
            display(widgets.HTML("<hr style='border-color:#37474f;'/>"))
            display(widgets.HTML("<h3>🧬 ONNX Model Weight Inspector</h3>"))
            display(onnx_panel)

        def on_load_click(b):
            m = market_dd.value
            s = strategy_dd.value
            if s == "None":
                s = None

            # Load data
            df = self.load_data(m, s)
            if df.empty:
                with plot_panel:
                    clear_output()
                    display(
                        widgets.HTML(
                            f"<p style='color:#ef5350;'>Failed to load data for {m}.</p>"
                        )
                    )
                return

            # Plot main charts
            with plot_panel:
                clear_output()
                fig = self.render_charts(df, m)
                fig_widget = go.FigureWidget(fig)
                display(fig_widget)

            # Step Inspector Setup
            with inspector_panel:
                clear_output()
                slider = widgets.IntSlider(
                    value=0,
                    min=0,
                    max=len(df) - 1,
                    step=1,
                    description="Simulation Step:",
                    layout=widgets.Layout(width="600px"),
                    continuous_update=False,
                )

                step_output = widgets.Output()

                def on_step_change(change):
                    idx = change["new"]
                    row = df.iloc[idx]
                    with step_output:
                        clear_output(wait=True)
                        self.inspect_step(row)

                slider.observe(on_step_change, names="value")
                display(slider)
                display(step_output)
                on_step_change({"new": 0})

            # ONNX Weights panel setup
            if onnx_model_path:
                with onnx_panel:
                    clear_output()
                    self.plot_onnx_weights(onnx_model_path)

        load_btn.on_click(on_load_click)
        on_load_click(None)
