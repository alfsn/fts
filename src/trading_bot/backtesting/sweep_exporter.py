# src/trading_bot/backtesting/sweep_exporter.py

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy.orm import Session

from trading_bot.backtesting.sweep_results import SweepResult
from trading_bot.backtesting.sweep_visualizer import SweepVisualizer
from trading_bot.config import settings


class BaseSweepExporter(ABC):
    """
    Abstract Base Class representing a parameter sweep visualization exporter.
    Adheres to the Open/Closed Principle to allow extending to new export formats.
    """

    @abstractmethod
    def export(
        self,
        sweep_result: SweepResult,
        output_path: Optional[str] = None,
        db_session: Optional[Session] = None,
    ) -> str:
        """
        Exports the parameter sweep visualization.

        :param sweep_result: Evaluated SweepResult object.
        :param output_path: Optional file path or directory to write the report to.
        :param db_session: Optional database session to query trial equity curves.
        :return: Absolute or relative file path of the generated visualization report.
        """
        pass


class HTMLSweepExporter(BaseSweepExporter):
    """
    Saves parameter sweep visualization charts as a standalone interactive HTML report.
    """

    def __init__(
        self,
        visualizer: Optional[SweepVisualizer] = None,
        db_url: str = "sqlite:///./dev.db",
    ) -> None:
        """
        Initializes the HTML sweep exporter. Supports dependency injection of the visualizer.
        """
        self.visualizer = visualizer or SweepVisualizer(db_url)
        self.logger = logging.getLogger(__name__)

    def export(
        self,
        sweep_result: SweepResult,
        output_path: Optional[str] = None,
        db_session: Optional[Session] = None,
    ) -> str:
        """
        Generates the interactive Plotly sweep charts and writes them out to a standalone HTML file.
        """
        if not sweep_result.trials:
            raise ValueError(
                f"Cannot export empty SweepResult: {sweep_result.sweep_name}"
            )

        fig = self.visualizer.render_charts(sweep_result, db_session=db_session)

        filename = f"sweep_report_{sweep_result.sweep_name}.html"

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
            target_dir = os.path.join(settings.RUNS_DIR, "reports")
            os.makedirs(target_dir, exist_ok=True)
            full_path = os.path.join(target_dir, filename)
        elif is_dir:
            os.makedirs(output_path, exist_ok=True)
            full_path = os.path.join(output_path, filename)
        else:
            parent_dir = os.path.dirname(output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            full_path = output_path

        self.logger.info(f"Saving interactive sweep visualization HTML to: {full_path}")

        import plotly.io as pio

        pio.write_html(
            fig,
            file=full_path,
            auto_open=False,
            include_plotlyjs="cdn",
        )

        return full_path
