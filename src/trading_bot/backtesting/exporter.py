# src/trading_bot/backtesting/exporter.py

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from trading_bot.backtesting.visualizer import BacktestVisualizer


class BaseBacktestExporter(ABC):
    """
    Abstract Base Class representing a backtest visualization exporter.
    Adheres to the Open/Closed Principle to allow extending to new export formats.
    """

    @abstractmethod
    def export(
        self,
        market_id: str,
        strategy_name: Optional[str] = None,
        run_id: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Exports the backtest visualization for a given market, strategy, and run.

        :param market_id: The market ID (e.g. 'BTC/USDT').
        :param strategy_name: The strategy name.
        :param run_id: The specific backtest simulation run ID.
        :param output_path: Optional file path or directory to write the report to.
        :return: Absolute or relative file path of the generated visualization report.
        """
        pass


class HTMLBacktestExporter(BaseBacktestExporter):
    """
    Saves the backtest visualization charts as a standalone interactive HTML file.
    """

    def __init__(
        self,
        visualizer: Optional[BacktestVisualizer] = None,
        db_url: str = "sqlite:///./dev.db",
    ) -> None:
        """
        Initializes the HTML exporter. Supports dependency injection of the visualizer.
        """
        self.visualizer = visualizer or BacktestVisualizer(db_url)
        self.logger = logging.getLogger(__name__)

    def export(
        self,
        market_id: str,
        strategy_name: Optional[str] = None,
        run_id: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Loads the backtest logs, generates the interactive Plotly charts,
        and writes them out to a standalone HTML file.
        """
        df = self.visualizer.load_data(market_id, strategy_name, run_id)
        if df.empty:
            raise ValueError(
                f"No backtest prediction data found for market: {market_id}, "
                f"strategy: {strategy_name}, run_id: {run_id}"
            )

        # Generate the interactive Plotly figure
        fig = self.visualizer.render_charts(df, market_id)

        # Sanitize market_id for use in filenames (e.g., replace '/' with '_')
        sanitized_market = market_id.replace("/", "_")
        filename = f"report_{sanitized_market}_{strategy_name or 'None'}_{run_id or 'All'}.html"

        # Determine if output_path is intended to be a directory
        is_dir = False
        if output_path is not None:
            if os.path.isdir(output_path):
                is_dir = True
            elif (
                output_path.endswith(os.sep)
                or output_path.endswith("/")
                or output_path.endswith("\\")
            ):
                is_dir = True
            else:
                _, ext = os.path.splitext(output_path)
                if not ext:
                    is_dir = True

        if output_path is None:
            # Default to runs/reports/ directory
            target_dir = os.path.join("runs", "reports")
            os.makedirs(target_dir, exist_ok=True)
            full_path = os.path.join(target_dir, filename)
        elif is_dir:
            os.makedirs(output_path, exist_ok=True)
            full_path = os.path.join(output_path, filename)
        else:
            # File path specified directly; ensure target directory exists
            parent_dir = os.path.dirname(output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            full_path = output_path

        self.logger.info(
            f"Saving interactive backtest visualization HTML to: {full_path}"
        )

        import plotly.io as pio

        pio.write_html(
            fig,
            file=full_path,
            auto_open=False,
            include_plotlyjs="cdn",  # Minimizes file size by using Plotly CDN
        )

        return full_path
