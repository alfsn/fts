"""M2-D3: one place resolves where the shared price cache lives."""

from pathlib import Path

import quant_data.db.cache as cache
from quant_data.db import cache_url


def _root():
    return next(
        p for p in Path(cache.__file__).resolve().parents if (p / "uv.lock").exists()
    )


def test_cache_url_override_then_env_then_workspace_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_CACHE_URL", "sqlite:////data/from-env.db")
    assert cache_url("sqlite:////data/override.db") == "sqlite:////data/override.db"
    assert cache_url() == "sqlite:////data/from-env.db"

    monkeypatch.delenv("MARKET_CACHE_URL")
    monkeypatch.chdir(tmp_path)  # the default doesn't depend on the working dir
    root = _root()
    assert (root / "packages" / "quant_data").is_dir()
    assert cache_url() == f"sqlite:///{root / 'market_cache.db'}"


def test_a_relative_sqlite_path_resolves_against_the_workspace_root(
    monkeypatch, tmp_path
):
    # Otherwise each app's working folder would get its own cache (PR 59 review).
    monkeypatch.chdir(tmp_path)
    root = _root()
    assert cache_url("sqlite:///./cache.db") == f"sqlite:///{root / 'cache.db'}"
    monkeypatch.setenv("MARKET_CACHE_URL", "sqlite+pysqlite:///data/cache.db")
    assert cache_url() == f"sqlite+pysqlite:///{root / 'data' / 'cache.db'}"
    # In-memory and non-SQLite URLs are left alone.
    assert cache_url("sqlite://") == "sqlite://"
    assert cache_url("postgresql://u@h/db") == "postgresql://u@h/db"
