"""M2-D3: one place resolves where the shared price cache lives."""

from pathlib import Path

import quant_data.db.cache as cache
from quant_data.db import cache_url


def test_cache_url_override_then_env_then_workspace_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_CACHE_URL", "sqlite:///from-env.db")
    assert cache_url("sqlite:///override.db") == "sqlite:///override.db"
    assert cache_url() == "sqlite:///from-env.db"

    monkeypatch.delenv("MARKET_CACHE_URL")
    monkeypatch.chdir(tmp_path)  # the default doesn't depend on the working dir
    root = next(
        p for p in Path(cache.__file__).resolve().parents if (p / "uv.lock").exists()
    )
    assert (root / "packages" / "quant_data").is_dir()
    assert cache_url() == f"sqlite:///{root / 'market_cache.db'}"
