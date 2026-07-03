# tests/test_catalog_repository.py

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_bot.core.catalog_repository import (
    BacktestCatalogRepository,
    CatalogQueryService,
    ModelCatalogRepository,
)
from trading_bot.core.database import Base
from trading_bot.core.enums import OrderSide, OrderStatus
from trading_bot.core.models import (
    BacktestEquityLog,
    BacktestPredictionLog,
    BarDataLog,
    Market,
    ModelRegistryLog,
    OrderLog,
    TradeLog,
)


@pytest.fixture
def db_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory


def test_model_catalog_repository_list_and_details(db_session_factory):
    repo = ModelCatalogRepository(db_session_factory)

    with db_session_factory() as session:
        m1 = ModelRegistryLog(
            model_id="model_lgb_001",
            run_id="run_101",
            model_type="LightGBM",
            market_id="AAPL",
            interval="1m",
            horizon=5,
            onnx_path="/tmp/model1.onnx",
            hyperparameters={"num_leaves": 31},
            metrics={"accuracy": 0.65, "f1": 0.62},
            status="candidate",
        )
        m2 = ModelRegistryLog(
            model_id="model_xgb_002",
            run_id="run_102",
            model_type="XGBoost",
            market_id="AAPL",
            interval="1m",
            horizon=5,
            onnx_path="/tmp/model2.onnx",
            hyperparameters={"max_depth": 6},
            metrics={"accuracy": 0.70, "f1": 0.68},
            status="production",
        )
        session.add_all([m1, m2])
        session.commit()

    # List all
    all_models = repo.list_models()
    assert len(all_models) == 2

    # Filter by status
    prod_models = repo.list_models(status="production")
    assert len(prod_models) == 1
    assert prod_models[0].model_id == "model_xgb_002"

    # Get details
    detail = repo.get_model_details("model_lgb_001")
    assert detail is not None
    assert detail.model_id == "model_lgb_001"
    assert detail.hyperparameters["num_leaves"] == 31


def test_model_promotion_demotes_prior_production(db_session_factory):
    repo = ModelCatalogRepository(db_session_factory)

    with db_session_factory() as session:
        m1 = ModelRegistryLog(
            model_id="model_lgb_001",
            model_type="LightGBM",
            market_id="AAPL",
            interval="1m",
            horizon=5,
            onnx_path="/tmp/model1.onnx",
            hyperparameters={},
            metrics={},
            status="production",
        )
        m2 = ModelRegistryLog(
            model_id="model_lgb_002",
            model_type="LightGBM",
            market_id="AAPL",
            interval="1m",
            horizon=5,
            onnx_path="/tmp/model2.onnx",
            hyperparameters={},
            metrics={},
            status="candidate",
        )
        session.add_all([m1, m2])
        session.commit()

    # Promote m2 to production
    success = repo.update_model_status("model_lgb_002", "production")
    assert success is True

    # Verify m1 is demoted to candidate and m2 is production
    m1_detail = repo.get_model_details("model_lgb_001")
    m2_detail = repo.get_model_details("model_lgb_002")

    assert m1_detail.status == "candidate"
    assert m2_detail.status == "production"


def test_backtest_catalog_repository_metrics(db_session_factory):
    repo = BacktestCatalogRepository(db_session_factory)
    now = datetime.now(timezone.utc)

    with db_session_factory() as session:
        # Populate backtest equity curve: 100 -> 110 -> 105 -> 120
        e1 = BacktestEquityLog(
            run_id="run_test",
            timestamp=now,
            cash=100.0,
            position=0.0,
            close=10.0,
            equity=100.0,
        )
        e2 = BacktestEquityLog(
            run_id="run_test",
            timestamp=now + timedelta(minutes=1),
            cash=110.0,
            position=0.0,
            close=11.0,
            equity=110.0,
        )
        e3 = BacktestEquityLog(
            run_id="run_test",
            timestamp=now + timedelta(minutes=2),
            cash=105.0,
            position=0.0,
            close=10.5,
            equity=105.0,
        )
        e4 = BacktestEquityLog(
            run_id="run_test",
            timestamp=now + timedelta(minutes=3),
            cash=120.0,
            position=0.0,
            close=12.0,
            equity=120.0,
        )

        # Order log sample
        o1 = OrderLog(
            order_id="ord_1",
            run_id="run_test",
            market_id="AAPL",
            strategy_name="TrendStrategy",
            side=OrderSide.BUY,
            requested_size=1.0,
            requested_price=10.0,
            status=OrderStatus.FILLED,
        )

        # Trade log sample
        t1 = TradeLog(
            order_id="ord_1",
            run_id="run_test",
            market_id="AAPL",
            side=OrderSide.BUY,
            fill_size=1.0,
            fill_price=10.0,
            outcome="win",
            fill_timestamp=now + timedelta(minutes=1),
        )

        session.add_all([e1, e2, e3, e4, o1, t1])
        session.commit()

    runs = repo.list_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run.run_id == "run_test"
    assert run.strategy_name == "TrendStrategy"
    assert run.market_id == "AAPL"
    assert run.total_return == 20.0  # (120 - 100)/100 * 100%
    assert run.max_drawdown > 0  # 110 -> 105 is DD
    assert run.total_trades == 1

    detail = repo.get_run_details("run_test")
    assert detail is not None
    assert len(detail.equity_curve) == 4
    assert len(detail.trades) == 1


def test_catalog_query_service_summary(db_session_factory):
    service = CatalogQueryService(db_session_factory)

    with db_session_factory() as session:
        m = Market(
            market_id="TSLA",
            name="Tesla Inc.",
            end_date=datetime.now(timezone.utc) + timedelta(days=365),
        )
        session.add(m)
        session.commit()

    summary = service.get_database_summary()
    assert summary["markets"] == 1
    assert summary["models"] == 0
