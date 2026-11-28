"""D10, D18: the reset drops only the market-data cache tables (markets included)."""

from quant_data.db import Base
from quant_data.db.reset import CACHE_TABLES, cache_row_counts, reset_market_cache
from sqlalchemy import create_engine, text


def _engine_with_data():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Things that are NOT a cache and must survive (fts dev.db has these).
        conn.execute(
            text("CREATE TABLE model_registry (id INTEGER PRIMARY KEY, name TEXT)")
        )
        conn.execute(text("INSERT INTO model_registry (name) VALUES ('rnn-v1')"))
        # A retired cache table left by an older schema (W4).
        conn.execute(text("CREATE TABLE fx_rates (id INTEGER PRIMARY KEY, rate REAL)"))
        conn.execute(text("INSERT INTO fx_rates (rate) VALUES (1000.0)"))
        conn.execute(
            text("INSERT INTO markets (instrument_id, name) VALUES ('AAPL', 'Apple')")
        )
        conn.execute(
            text(
                "INSERT INTO bar_data_logs (instrument_id, timestamp, open, high, low, close, "
                "volume, bar_type, interval, ticks_count, dollar_volume, source) VALUES "
                "('AAPL', '2024-01-02 00:00:00', 1, 1, 1, 1, 0, 'TIME', '1d', 1, 0, 'yahoo')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO corporate_actions (instrument_id, action_type, ex_date, ratio, source) "
                "VALUES ('AAPL', 'SPLIT', '2020-08-31 00:00:00', 4.0, 'yahoo')"
            )
        )
    return engine


def test_reset_drops_only_the_cache_tables():
    engine = _engine_with_data()
    assert [t.name for t in CACHE_TABLES] == [
        "markets",
        "bar_data_logs",
        "corporate_actions",
    ]
    assert cache_row_counts(engine) == {
        "markets": 1,
        "bar_data_logs": 1,
        "corporate_actions": 1,
        "fx_rates": 1,
    }

    reset_market_cache(engine)

    assert cache_row_counts(engine) == {
        "markets": 0,
        "bar_data_logs": 0,
        "corporate_actions": 0,
    }
    with engine.connect() as conn:
        assert (
            conn.execute(text("SELECT name FROM model_registry")).scalar() == "rnn-v1"
        )
        # Recreated with the new schema.
        cols = {
            row[1] for row in conn.execute(text("PRAGMA table_info(bar_data_logs)"))
        }
        assert "source" in cols


def test_reset_tolerates_a_db_without_the_tables():
    engine = create_engine("sqlite:///:memory:")
    assert cache_row_counts(engine) == {
        "markets": 0,
        "bar_data_logs": 0,
        "corporate_actions": 0,
    }
    reset_market_cache(engine)
    assert cache_row_counts(engine) == {
        "markets": 0,
        "bar_data_logs": 0,
        "corporate_actions": 0,
    }
