# tests/unit/test_backtest_exporter.py

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from trading_bot.backtesting.exporter import HTMLBacktestExporter
from trading_bot.core.database import Base, SessionLocal
from trading_bot.core.database import engine as dev_engine
from trading_bot.core.enums import BarType
from trading_bot.core.models import BacktestPredictionLog, BarDataLog, Market


@pytest.fixture
def db_session():
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)

    with patch("trading_bot.core.database.engine", test_engine):
        SessionLocal.configure(bind=test_engine)
        db = SessionLocal()

        market = Market(
            market_id="BTC/USDT",
            name="Bitcoin / Tether",
            end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
            resolution_source="Binance",
        )
        db.add(market)

        # Create 5 bars
        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        for i in range(5):
            bar = BarDataLog(
                market_id="BTC/USDT",
                timestamp=base_time + timedelta(minutes=i * 5),
                open=40000.0 + i * 10,
                high=40050.0 + i * 10,
                low=39980.0 + i * 10,
                close=40010.0 + i * 10,
                volume=1.5,
                bar_type=BarType.TIME,
                ticks_count=50,
                dollar_volume=60000.0,
            )
            db.add(bar)

            # Create 5 prediction logs
            pred = BacktestPredictionLog(
                market_id="BTC/USDT",
                timestamp=base_time + timedelta(minutes=i * 5),
                strategy_name="cnn",
                prediction_output="[0.1, 0.2, 0.7]",
                predicted_signal="BUY" if i % 2 == 0 else "HOLD",
                confidence=0.8,
                actual_future_return=0.01,
                run_id="test_run_123",
            )
            db.add(pred)

        db.commit()
        yield db
        db.close()
        SessionLocal.configure(bind=dev_engine)


def test_html_exporter(db_session: Session, tmp_path):
    exporter = HTMLBacktestExporter(db_url="sqlite:///:memory:")
    # Inject our active test session into the visualizer SessionLocal config
    exporter.visualizer.SessionLocal = lambda: db_session

    output_file = tmp_path / "report.html"

    result_path = exporter.export(
        market_id="BTC/USDT",
        strategy_name="cnn",
        run_id="test_run_123",
        output_path=str(output_file),
    )

    assert result_path == str(output_file)
    assert os.path.exists(result_path)

    # Read the output and check if it contains plotly details
    with open(result_path, "r") as f:
        html_content = f.read()
        assert "Plotly" in html_content or "plotly" in html_content.lower()
        assert "BTC" in html_content
        assert "USDT" in html_content
        assert "ML Strategy" in html_content


def test_html_exporter_directory_output(db_session: Session, tmp_path):
    exporter = HTMLBacktestExporter(db_url="sqlite:///:memory:")
    exporter.visualizer.SessionLocal = lambda: db_session

    output_dir = tmp_path / "test_reports"
    result_path = exporter.export(
        market_id="BTC/USDT",
        strategy_name="cnn",
        run_id="test_run_123",
        output_path=str(output_dir),
    )

    assert os.path.exists(result_path)
    assert result_path.startswith(str(output_dir))
    assert result_path.endswith(".html")


def test_html_exporter_empty_dataframe_raises(db_session: Session):
    exporter = HTMLBacktestExporter(db_url="sqlite:///:memory:")
    exporter.visualizer.SessionLocal = lambda: db_session

    with pytest.raises(ValueError, match="No backtest prediction data found"):
        # Querying market that has no logs raises
        exporter.export(
            market_id="NONEXISTENT_MARKET",
            strategy_name="cnn",
            run_id="test_run_123",
        )
