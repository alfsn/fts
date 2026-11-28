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
from quant_core.analytics import wealth_index
from quant_core.enums import OrderSide, SignalType
from quant_core.intervals import interval_to_timedelta
from quant_data.db import MarketDataRepository, cache_url, init_cache
from sqlalchemy.orm import sessionmaker
from trading_bot.config import get_settings
from trading_bot.core.database import create_db_engine
from trading_bot.runs import RunStore

logger = logging.getLogger(__name__)


def _cumulative_return(levels: pd.Series) -> pd.Series:
    """Growth of 1 minus 1 over the priced rows (a non-positive level is
    missing), compounding across gaps; NaN where unpriced or undefined."""
    priced = levels.where(levels > 0).dropna()
    if priced.empty:
        return pd.Series(np.nan, index=levels.index)
    return wealth_index(priced).reindex(levels.index) - 1.0


class BacktestVisualizer:
    """
    Provides interactive widgets and Plotly charts for Jupyter Notebooks
    to inspect model outputs, backtest P&L, risk/sizing decisions,
    and neural network filters/weights.
    """

    def __init__(
        self, db_url: Optional[str] = None, cache_db_url: Optional[str] = None
    ) -> None:
        """
        Initializes the visualizer on the research database (runs, D16) and
        the shared price cache (bars, D18).
        """
        settings = get_settings()
        self.engine = create_db_engine(db_url or settings.DATABASE_URL)
        self.SessionLocal = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False
        )
        self.store = RunStore(self.SessionLocal)
        cache_engine = create_db_engine(
            cache_url(cache_db_url or settings.MARKET_CACHE_URL)
        )
        init_cache(cache_engine)
        self.CacheSessionLocal = sessionmaker(
            bind=cache_engine, autocommit=False, autoflush=False
        )

    def intervals(self, instrument_id: str) -> List[str]:
        """The intervals cached for one market, for the interval picker."""
        with self.CacheSessionLocal() as session:
            return MarketDataRepository(session).intervals(instrument_id)

    def run_interval(self, instrument_id: str, run_id: Optional[str]) -> Optional[str]:
        """The cached interval a run traded on, read from the spacing of its
        predictions for this market; None when no cached interval matches."""
        if not run_id or run_id == "None":
            return None
        record = self.store.load(run_id)
        if record is None:
            return None
        return self._interval_of(record, instrument_id)

    def _interval_of(self, record, instrument_id: str) -> Optional[str]:
        stamps = pd.Series(
            sorted(
                {
                    p.timestamp
                    for p in record.predictions
                    if p.instrument_id == instrument_id
                }
            )
        )
        if len(stamps) < 2:
            return None
        spacing = pd.to_datetime(stamps, utc=True).diff().median()
        for interval in self.intervals(instrument_id):
            try:
                if interval_to_timedelta(interval) == spacing:
                    return interval
            except ValueError:
                continue
        return None

    def _bars(self, instrument_id: str, interval: Optional[str]) -> pd.DataFrame:
        """One cached series as a frame, through ``quant_data`` (M2a). Without
        an interval, the market's only one; two cached intervals never mix."""
        if interval is None:
            cached = self.intervals(instrument_id)
            if len(cached) > 1:
                raise ValueError(
                    f"{instrument_id} is cached at {cached}; pick an interval."
                )
            if not cached:
                return pd.DataFrame()
            interval = cached[0]
        with self.CacheSessionLocal() as session:
            bars = MarketDataRepository(session).load_bars(instrument_id, interval)
        return pd.DataFrame(
            [
                b.model_dump(
                    include={
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "dollar_volume",
                    }
                )
                for b in bars
            ]
        )

    def load_data(
        self,
        instrument_id: str,
        strategy_name: Optional[str] = None,
        run_id: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Joins the instrument's bars (one interval; see ``_bars``) with one
        run's predictions, fills and equity (from ``RunStore``). Without a
        run, the bars only. Without an interval, the run's own (see
        ``run_interval``).
        """
        if strategy_name == "None":
            strategy_name = None
        if run_id == "None":
            run_id = None

        record = self.store.load(run_id) if run_id else None
        if interval is None and record is not None:
            interval = self._interval_of(record, instrument_id)

        df_bars = self._bars(instrument_id, interval)
        if df_bars.empty:
            logger.warning(f"No bar data found for market: {instrument_id}")
            return pd.DataFrame()
        df_bars["timestamp"] = pd.to_datetime(df_bars["timestamp"], utc=True)

        preds = [
            p.model_dump()
            for p in (record.predictions if record else [])
            if p.instrument_id == instrument_id
            and (not strategy_name or p.strategy_name == strategy_name)
        ]
        if preds:
            df_preds = pd.DataFrame(preds).drop(columns=["instrument_id"])
            df_preds["timestamp"] = pd.to_datetime(df_preds["timestamp"], utc=True)
            df = pd.merge(df_bars, df_preds, on="timestamp", how="left")
        else:
            df = df_bars.copy()
            for col in (
                "strategy_name",
                "prediction_output",
                "predicted_signal",
                "confidence",
                "actual_future_return",
            ):
                df[col] = None

        # Fills at their bar time (M1: the wall-clock order update time is gone)
        fills = [
            {
                "timestamp": f.fill_timestamp,
                "side": f.side.value.upper(),
                "quantity": f.quantity,
                "price": f.fill_price,
            }
            for f in (record.fills if record else [])
            if f.instrument_id == instrument_id
        ]
        if fills:
            df_trades = pd.DataFrame(fills)
            df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], utc=True)
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                df_trades.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("1h"),
            )
        else:
            df["side"] = None
            df["size"] = None
            df["price"] = None

        # Fill NaN values for prediction fields with default values to ensure clean downstream contracts
        if "predicted_signal" in df.columns:
            df["predicted_signal"] = df["predicted_signal"].fillna("None").astype(str)
        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].fillna(0.0).astype(float)

        if record and record.equity:
            df_equity = pd.DataFrame(
                [
                    {
                        "timestamp": e.timestamp,
                        "cash": e.cash,
                        "equity_position": e.position,
                        "equity_close": e.close,
                        "actual_equity": e.equity,
                    }
                    for e in record.equity
                ]
            )
            df_equity["timestamp"] = pd.to_datetime(df_equity["timestamp"], utc=True)
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                df_equity.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("1h"),
            )

        return df.sort_values("timestamp").reset_index(drop=True)

    def get_available_runs(
        self, instrument_id: str, strategy_name: Optional[str] = None
    ) -> List[str]:
        """Backtest run_ids that predicted this market (and strategy),
        newest first."""
        if strategy_name == "None":
            strategy_name = None
        run_ids: List[str] = []
        for m in self.store.markets(kind="backtest"):
            if m.instrument_id != instrument_id:
                continue
            if strategy_name and m.strategy_name != strategy_name:
                continue
            if m.run_id not in run_ids:
                run_ids.append(m.run_id)
        return run_ids

    def get_available_markets_and_strategies(self) -> Dict[str, List[str]]:
        """Markets backtest runs predicted and their strategies; without runs,
        the cached markets."""
        result: Dict[str, List[str]] = {}
        for m in self.store.markets(kind="backtest"):
            names = result.setdefault(m.instrument_id, [])
            if m.strategy_name not in names:
                names.append(m.strategy_name)
        if result:
            return result
        with self.CacheSessionLocal() as session:
            cached = MarketDataRepository(session).instruments()
        return {m: ["None"] for m in cached}

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

        # Cumulative P&L (buy-and-hold baseline and logged equity) from quant_core
        df["cum_baseline_return"] = _cumulative_return(df["close"])

        if "actual_equity" in df.columns and not df["actual_equity"].isnull().all():
            df["cum_strat_return"] = _cumulative_return(df["actual_equity"])

            if "equity_position" in df.columns:
                df["strat_position"] = df["equity_position"]
        else:
            # Fall back to signal-based simulated returns
            df["strat_position"] = df["position"].shift(1).fillna(0.0)
            df["strat_return"] = df["strat_position"] * df["raw_return"]
            df["cum_strat_return"] = (1.0 + df["strat_return"]).cumprod() - 1.0

        return df

    def render_charts(self, df: pd.DataFrame, instrument_id: str) -> go.Figure:
        """
        Generates a Plotly subplot showing:
        1. Candlestick chart with trade overlays.
        2. Cumulative returns (Strategy vs Baseline).
        """
        from quant_ui.plots.performance import plot_backtest_performance

        df_pnl = self.calculate_pnl(df)
        return plot_backtest_performance(df_pnl, instrument_id)

    def inspect_step(self, row: pd.Series) -> None:
        """
        Renders HTML table outputs for inspecting the model predictions (outputs),
        and risk manager sizing details for a specific step.
        """
        timestamp = row["timestamp"]
        close_price = row["close"]

        # Parse predictions
        predictions_str = row.get("prediction_output")
        if pd.isna(predictions_str) or predictions_str is None:
            predictions_str = None

        signal = str(row.get("predicted_signal", "None"))
        confidence = float(row.get("confidence", 0.0))

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

    def _default_interval(
        self, instrument_id: str, run_id: Optional[str]
    ) -> Optional[str]:
        """The picker's default: the run's interval, else the first cached."""
        cached = self.intervals(instrument_id)
        return self.run_interval(instrument_id, run_id) or (
            cached[0] if cached else None
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
        default_market = markets[0]
        default_strategies = meta_dict[default_market]
        default_strategy = default_strategies[0] if default_strategies else None

        # 2. Controls
        market_dd = widgets.Dropdown(
            options=markets,
            description="Market:",
            layout=widgets.Layout(width="200px"),
        )
        strategy_dd = widgets.Dropdown(
            options=default_strategies,
            description="Strategy:",
            layout=widgets.Layout(width="250px"),
        )
        run_ids = self.get_available_runs(default_market, default_strategy)
        run_id_dd = widgets.Dropdown(
            options=run_ids,
            value=run_ids[0] if run_ids else None,
            description="Run ID:",
            layout=widgets.Layout(width="220px"),
        )
        default_intervals = self.intervals(default_market)
        interval_dd = widgets.Dropdown(
            options=default_intervals,
            value=self._default_interval(default_market, run_id_dd.value),
            description="Interval:",
            layout=widgets.Layout(width="160px"),
        )

        load_btn = widgets.Button(
            description="Load Simulation Data",
            button_style="primary",
            layout=widgets.Layout(width="180px"),
        )

        control_box = widgets.HBox(
            [market_dd, interval_dd, strategy_dd, run_id_dd, load_btn]
        )
        display(control_box)

        def on_market_change(change):
            m = change["new"]
            interval_dd.options = self.intervals(m)
            strategies = meta_dict.get(m, ["None"])
            strategy_dd.options = strategies
            s = strategies[0] if strategies else None
            runs = self.get_available_runs(m, s if s != "None" else None)
            run_id_dd.options = runs
            run_id_dd.value = runs[0] if runs else None
            # The same run can trade several markets, so set it here too.
            interval_dd.value = self._default_interval(m, run_id_dd.value)

        def on_strategy_change(change):
            s = change["new"]
            m = market_dd.value
            runs = self.get_available_runs(m, s if s != "None" else None)
            run_id_dd.options = runs
            run_id_dd.value = runs[0] if runs else None

        def on_run_change(change):
            interval_dd.value = self._default_interval(market_dd.value, change["new"])

        market_dd.observe(on_market_change, names="value")
        strategy_dd.observe(on_strategy_change, names="value")
        run_id_dd.observe(on_run_change, names="value")

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

            r = run_id_dd.value
            if not r:
                with plot_panel:
                    clear_output()
                    display(
                        widgets.HTML(
                            "<p style='color:#ef5350;'>No Run ID available or selected.</p>"
                        )
                    )
                return

            # Load data
            df = self.load_data(m, s, r, interval=interval_dd.value)
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
