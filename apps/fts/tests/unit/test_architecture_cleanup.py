# tests/unit/test_architecture_cleanup.py

import os
import re
from pathlib import Path

import pytest
from quant_core.enums import OrderSide, OrderStatus
from quant_core.models import ExecutionResult, OrderRequest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trading_bot.core.database import ResearchBase, create_db_session, init_db
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.runs import RunHeader, SqlRunRecorder
from trading_bot.runs.tables import (
    BacktestPredictionLog,
    ModelPredictionLog,
    PredictionLog,
    RunLog,
    TradeLog,
)


def test_dynamic_db_session_and_init():
    """Verifies that create_db_session connects dynamically and init_db runs on custom bind."""
    test_db_file = "./temp_test_isolation.db"
    if os.path.exists(test_db_file):
        os.remove(test_db_file)

    try:
        db_url = f"sqlite+pysqlite:///{test_db_file}"
        session = create_db_session(db_url)

        # Verify custom engine and session binding
        assert session.bind.url.database == test_db_file

        # Initialize tables on custom bind
        init_db(bind_engine=session.bind)

        # Check table creation by running simple query
        assert session.query(RunLog).count() == 0
        session.close()
    finally:
        if os.path.exists(test_db_file):
            os.remove(test_db_file)


def test_prediction_log_single_table_inheritance():
    """Verifies that ModelPredictionLog and BacktestPredictionLog share the same table via STI."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    ResearchBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        from datetime import datetime, timezone

        # Create a live prediction log
        live_log = ModelPredictionLog(
            run_id="live_run_1",
            timestamp=datetime.now(timezone.utc),
            strategy_name="nets_strategy_cnn",
            instrument_id="AAPL",
            predicted_signal="buy",
            confidence=0.9,
        )

        # Create a backtest prediction log
        backtest_log = BacktestPredictionLog(
            run_id="backtest_run_1",
            timestamp=datetime.now(timezone.utc),
            strategy_name="nets_strategy_cnn",
            instrument_id="AAPL",
            predicted_signal="sell",
            confidence=0.75,
        )

        db.add(live_log)
        db.add(backtest_log)
        db.commit()

        # Both classes query the SAME physical table "prediction_logs"
        assert live_log.__tablename__ == "prediction_logs"
        assert backtest_log.__tablename__ == "prediction_logs"

        # Check polymorphic behavior
        # ModelPredictionLog query should only return live logs (identity = live)
        live_records = db.query(ModelPredictionLog).all()
        assert len(live_records) == 1
        assert live_records[0].run_id == "live_run_1"
        assert live_records[0].log_type == "live"

        # BacktestPredictionLog query should only return backtest logs (identity = backtest)
        backtest_records = db.query(BacktestPredictionLog).all()
        assert len(backtest_records) == 1
        assert backtest_records[0].run_id == "backtest_run_1"
        assert backtest_records[0].log_type == "backtest"

        # Querying the base PredictionLog should return both
        all_records = db.query(PredictionLog).all()
        assert len(all_records) == 2
    finally:
        db.close()
        ResearchBase.metadata.drop_all(engine)


def test_portfolio_public_property_accessors():
    """Verifies that Portfolio exposes public properties instead of private attributes."""
    portfolio = Portfolio(initial_balance=5000.0, quote_currency="USD")

    # Verify public cash balance property
    assert portfolio.cash_balance == 5000.0

    # Verify public positions and open_orders properties
    assert isinstance(portfolio.positions, dict)
    assert isinstance(portfolio.open_orders, dict)


def test_the_run_recorder_never_commits():
    """The recorder writes into the caller's transaction; the loop or the
    execution engine decides when to commit."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    ResearchBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        from datetime import datetime, timezone

        recorder = SqlRunRecorder(run_id="test_run", kind="backtest")
        order = OrderRequest(
            instrument_id="AAPL", side=OrderSide.BUY, quantity=10.0, price=150.0
        )
        result = ExecutionResult(
            order_id="o1",
            status=OrderStatus.FILLED,
            filled_quantity=10.0,
            avg_price=150.0,
            timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

        recorder.start(db, RunHeader(run_id="test_run", kind="backtest"))
        recorder.record_fill(db, order, result)
        assert db.query(TradeLog).count() == 1

        db.rollback()
        assert db.query(TradeLog).count() == 0
        assert db.query(RunLog).count() == 0
    finally:
        db.close()
        ResearchBase.metadata.drop_all(engine)


# --- Text guards (M1, M2b, R1; the import rules are import-linter contracts) ---

REPO = Path(__file__).resolve().parents[4]
APP_SOURCES = (
    REPO / "apps" / "fts" / "src" / "trading_bot",
    REPO / "apps" / "fts" / "src" / "plugins",
    REPO / "apps" / "portfolio_analyser" / "src",
)
CACHE_MODULE = (
    REPO / "packages" / "quant_data" / "src" / "quant_data" / "db" / "cache.py"
)


def _source_hits(pattern: str, roots=APP_SOURCES):
    rx = re.compile(pattern)
    return [
        f"{p.relative_to(REPO)}:{i}"
        for root in roots
        for p in root.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if rx.search(line)
    ]


def test_one_database_no_backtest_url_or_test_switch():
    """D16: no second backtest database and no "test"-in-URL switch."""
    assert _source_hits(r"BACKTEST_DATABASE_URL") == []
    assert _source_hits(r"""["']test["']\s+in\s+\w*url""") == []


def test_positions_are_not_stored():
    """D19: positions are derived from fills; nothing persists them."""
    assert _source_hits(r"\bPositionRepository\b|__tablename__ = \"positions\"") == []


def test_ic_maths_only_in_quant_core():
    """R1: correlations (the IC and its SE) are computed only in
    ``quant_core.analytics.forecast``, never in fts."""
    pattern = r"\bcorrcoef\b|spearmanr|pearsonr|\.corr\(|\.corrwith\("
    assert _source_hits(pattern, APP_SOURCES[:2]) == []  # trading_bot, plugins


def test_run_reset_drops_only_run_tables(tmp_path):
    """The reset drops the run tables and the retired ``positions`` table,
    never the model registry. The research DB holds no cache tables (D18)."""
    from sqlalchemy import inspect, text
    from trading_bot.runs.reset import reset_runs
    from trading_bot.runs.tables import RUN_TABLES

    engine = create_engine(f"sqlite:///{tmp_path / 'research.db'}")
    init_db(bind_engine=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE positions (id INTEGER PRIMARY KEY, instrument_id "
                "VARCHAR REFERENCES markets (instrument_id))"
            )
        )

    reset_runs(engine)

    tables = set(inspect(engine).get_table_names())
    assert "positions" not in tables
    assert "model_registry" in tables
    assert not {"markets", "bar_data_logs", "corporate_actions", "fx_rates"} & tables
    assert {t.name for t in RUN_TABLES} <= tables


def test_one_cache_file_resolved_only_by_quant_data():
    """D18, M2-D3: no app names a cache file or the old analyser variable, and
    only ``quant_data.db.cache`` gives ``MARKET_CACHE_URL`` a default."""
    assert (
        _source_hits(r"market_cache\.db|portfolio_analyser\.db|PA_DATABASE_URL") == []
    )
    default = (
        r"""["']MARKET_CACHE_URL["']\s*,\s*(?!default\s*=\s*None)\S"""
        r"|MARKET_CACHE_URL\s*:[^=\n]*=\s*(?!None\b)\S"
    )
    package_sources = [
        p / "src" for p in (REPO / "packages").iterdir() if (p / "src").is_dir()
    ]
    hits = _source_hits(default, APP_SOURCES + tuple(package_sources))
    assert [
        h for h in hits if not h.startswith(str(CACHE_MODULE.relative_to(REPO)))
    ] == []


def test_fts_refuses_a_cache_url_in_its_env_file(monkeypatch, tmp_path):
    """The analyser never reads fts's .env, so a cache URL there would give
    the two apps different caches (PR 59 review): fail loudly instead."""
    import pytest
    from trading_bot.config import Settings

    env_file = Settings.model_config["env_file"]
    (tmp_path / env_file).write_text('MARKET_CACHE_URL="sqlite:////x/cache.db"\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Export it in the shell"):
        Settings()
