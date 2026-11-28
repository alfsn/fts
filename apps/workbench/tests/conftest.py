"""Each page runs against temp databases: one fts research DB holding a model
and a recorded run, and an empty price cache."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import sessionmaker
from trading_bot.core.catalog_repository import CatalogQueryService
from trading_bot.core.models import ModelRegistryLog
from trading_bot.runs.tables import BacktestEquityLog, RunLog, RunSegmentLog

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def databases(tmp_path, monkeypatch):
    research = f"sqlite:///{tmp_path / 'research.db'}"
    monkeypatch.setenv("DATABASE_URL", research)
    monkeypatch.setenv("MARKET_CACHE_URL", f"sqlite:///{tmp_path / 'cache.db'}")
    catalog = CatalogQueryService.open(research)  # creates the tables
    with catalog.session_factory() as session:
        session.add_all(
            [
                ModelRegistryLog(
                    model_id="model_1",
                    model_type="LSTM",
                    instrument_id="BTC/USDT",
                    interval="1h",
                    horizon=1,
                    onnx_path="/nonexistent.onnx",
                    hyperparameters={"hidden_dim": 64, "lr": float("nan")},
                    metrics={"val_ic": 0.03, "val_ic_se": 0.01, "val_ic_n": 100},
                    status="candidate",
                ),
                RunLog(run_id="run_1", kind="backtest", instrument_id="BTC/USDT"),
                RunSegmentLog(run_id="run_1", model_id="model_1"),
            ]
            + [
                BacktestEquityLog(
                    run_id="run_1",
                    timestamp=T0 + timedelta(hours=i),
                    cash=equity,
                    position=0.0,
                    close=100.0 + i,
                    equity=equity,
                )
                for i, equity in enumerate([1000.0, 1010.0, 990.0, 1030.0])
            ]
        )
        session.commit()
    return catalog
