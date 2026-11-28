# tests/unit/test_backtest_visualizer.py

from datetime import datetime, timedelta, timezone

import pytest
from quant_core.enums import BarType
from quant_data.db import BarDataLog, Market, init_cache
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trading_bot.backtesting.visualizer import BacktestVisualizer
from trading_bot.core.database import ResearchBase
from trading_bot.runs.tables import BacktestEquityLog, BacktestPredictionLog, RunLog


@pytest.fixture
def db_url(tmp_path):
    """One file holding one run (research tables) and its bars (cache tables)."""
    url = f"sqlite:///{tmp_path / 'research.db'}"
    test_engine = create_engine(url)
    ResearchBase.metadata.create_all(bind=test_engine)
    init_cache(test_engine)

    with sessionmaker(bind=test_engine)() as db:
        db.add(
            RunLog(
                run_id="run_123",
                kind="backtest",
                strategy_name="cnn",
                instrument_id="BTC/USDT",
            )
        )

        market = Market(
            instrument_id="BTC/USDT",
            name="Bitcoin / Tether",
            end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
            resolution_source="Binance",
        )
        db.add(market)

        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        for i in range(5):
            bar = BarDataLog(
                interval="5m",
                source="test",
                instrument_id="BTC/USDT",
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

            pred = BacktestPredictionLog(
                instrument_id="BTC/USDT",
                timestamp=base_time + timedelta(minutes=i * 5),
                strategy_name="cnn",
                prediction_output="[0.1, 0.2, 0.7]",
                predicted_signal="BUY" if i % 2 == 0 else "HOLD",
                confidence=0.8,
                actual_future_return=0.01,
                run_id="run_123",
            )
            db.add(pred)

            equity = BacktestEquityLog(
                run_id="run_123",
                timestamp=base_time + timedelta(minutes=i * 5),
                cash=10000.0,
                position=1.0,
                close=40010.0 + i * 10,
                equity=10000.0 + i * 100.0,
            )
            db.add(equity)

        db.commit()
    yield url
    test_engine.dispose()


def test_backtest_visualizer_load_data_with_none_run_id(db_url):
    viz = BacktestVisualizer(db_url=db_url, cache_db_url=db_url)

    # When run_id is None, load_data still loads the bars (no run to join)
    df = viz.load_data(instrument_id="BTC/USDT", strategy_name="cnn", run_id=None)
    assert not df.empty
    assert len(df) == 5
    assert "close" in df.columns
    assert "predicted_signal" in df.columns


def test_backtest_visualizer_load_data_with_specific_run_id(db_url):
    viz = BacktestVisualizer(db_url=db_url, cache_db_url=db_url)

    df = viz.load_data(instrument_id="BTC/USDT", strategy_name="cnn", run_id="run_123")
    assert not df.empty
    assert len(df) == 5
    assert "actual_equity" in df.columns
    assert (df["predicted_signal"] != "None").all()


def test_backtest_visualizer_get_available_runs(db_url):
    viz = BacktestVisualizer(db_url=db_url, cache_db_url=db_url)

    runs = viz.get_available_runs(instrument_id="BTC/USDT", strategy_name="None")
    assert runs == ["run_123"]


def test_the_picker_lists_a_run_under_every_market_and_strategy_it_predicted(
    db_url,
):
    """The card names one instrument and the first strategy; the picker reads
    the predictions, so a second market or strategy is not hidden."""
    engine = create_engine(db_url)
    t = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    with sessionmaker(bind=engine)() as db:
        db.add(Market(instrument_id="ETH/USDT", name="Ether / Tether"))
        for iid, strat in [("BTC/USDT", "lstm"), ("ETH/USDT", "cnn")]:
            db.add(
                BacktestPredictionLog(
                    instrument_id=iid,
                    timestamp=t,
                    strategy_name=strat,
                    predicted_signal="BUY",
                    confidence=0.5,
                    run_id="run_123",
                )
            )
        db.commit()
    engine.dispose()

    viz = BacktestVisualizer(db_url=db_url, cache_db_url=db_url)
    assert viz.get_available_markets_and_strategies() == {
        "BTC/USDT": ["cnn", "lstm"],
        "ETH/USDT": ["cnn"],
    }
    assert viz.get_available_runs("ETH/USDT", "cnn") == ["run_123"]
    assert viz.get_available_runs("BTC/USDT", "lstm") == ["run_123"]


def test_calculate_pnl_compounds_across_gaps_and_is_nan_when_undefined():
    import numpy as np
    import pandas as pd

    viz = BacktestVisualizer(db_url="sqlite:///:memory:")
    df = pd.DataFrame(
        {
            "close": [100.0, 0.0, 110.0, 121.0],  # 0.0: the engine had no bar
            "actual_equity": [1000.0, np.nan, 1100.0, 1210.0],
        }
    )

    pnl = viz.calculate_pnl(df)

    assert list(pnl["cum_baseline_return"].dropna()) == pytest.approx([0.0, 0.1, 0.21])
    assert np.isnan(pnl["cum_baseline_return"].iloc[1])
    assert list(pnl["cum_strat_return"].dropna()) == pytest.approx([0.0, 0.1, 0.21])

    broke = viz.calculate_pnl(df.assign(actual_equity=[0.0, 0.0, 0.0, 0.0]))
    assert broke["cum_strat_return"].isna().all()  # undefined, not 0.0
