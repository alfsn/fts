from trading_bot.dashboard.cli import _as_url


def test_as_url_keeps_urls_with_a_driver_and_wraps_bare_paths():
    # settings.DATABASE_URL and .env use sqlite+pysqlite (PR 59 review).
    assert _as_url("sqlite+pysqlite:///x.db") == "sqlite+pysqlite:///x.db"
    assert _as_url("postgresql+psycopg2://u@h/db") == "postgresql+psycopg2://u@h/db"
    assert _as_url("data/cache.db") == "sqlite:///data/cache.db"
