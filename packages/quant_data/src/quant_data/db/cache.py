"""
Where the shared price cache lives (plan D18, M2-D3).

Both apps (fts and portfolio_analyser) read bars, corporate actions and FX
rates from one file. This module holds the one default for its location:
``market_cache.db`` in the workspace root (the first parent folder holding
``uv.lock``). ``MARKET_CACHE_URL`` overrides it. A relative SQLite path
resolves against the workspace root, never the working folder, so every app
and folder opens the same file.
"""

import os
from pathlib import Path

from sqlalchemy.engine import Engine, make_url

from .models import BarDataLog, Base, CorporateActionRecord, Market

ENV_VAR = "MARKET_CACHE_URL"

# Parents before children, so drop_all / create_all order is right.
CACHE_TABLES = (
    Market.__table__,
    BarDataLog.__table__,
    CorporateActionRecord.__table__,
)


def _workspace_root() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "uv.lock").exists():
            return folder
    raise RuntimeError(
        f"No uv.lock above {__file__}: cannot place market_cache.db. Set {ENV_VAR}."
    )


def _rooted(url: str) -> str:
    """A relative SQLite path made absolute against the workspace root."""
    parsed = make_url(url)
    path = parsed.database
    if parsed.get_backend_name() != "sqlite" or not path or path == ":memory:":
        return url
    if Path(path).is_absolute():
        return url
    absolute = (_workspace_root() / path).resolve()
    return parsed.set(database=str(absolute)).render_as_string(hide_password=False)


def cache_url(override: str | None = None) -> str:
    """The cache's SQLAlchemy URL: ``override``, else ``$MARKET_CACHE_URL``,
    else ``sqlite:///<workspace root>/market_cache.db``. A relative SQLite
    path is resolved against the workspace root."""
    if override:
        return _rooted(override)
    if os.environ.get(ENV_VAR):
        return _rooted(os.environ[ENV_VAR])
    return f"sqlite:///{_workspace_root() / 'market_cache.db'}"


def init_cache(engine: Engine) -> None:
    """Create the cache tables (only those) if they don't exist yet."""
    Base.metadata.create_all(engine, tables=list(CACHE_TABLES))
