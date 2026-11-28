"""
Where the shared price cache lives (plan D18, M2-D3).

Both apps (fts and portfolio_analyser) read bars, corporate actions and FX
rates from one file. This module holds the one default for its location:
``market_cache.db`` in the workspace root (the first parent folder holding
``uv.lock``). ``MARKET_CACHE_URL`` overrides it.
"""

import os
from pathlib import Path

from sqlalchemy.engine import Engine

from .models import BarDataLog, Base, CorporateActionRecord, FXRateRecord, Market

ENV_VAR = "MARKET_CACHE_URL"

# Parents before children, so drop_all / create_all order is right.
CACHE_TABLES = (
    Market.__table__,
    BarDataLog.__table__,
    CorporateActionRecord.__table__,
    FXRateRecord.__table__,
)


def _workspace_root() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "uv.lock").exists():
            return folder
    raise RuntimeError(
        f"No uv.lock above {__file__}: cannot place market_cache.db. Set {ENV_VAR}."
    )


def cache_url(override: str | None = None) -> str:
    """The cache's SQLAlchemy URL: ``override``, else ``$MARKET_CACHE_URL``,
    else ``sqlite:///<workspace root>/market_cache.db``."""
    if override:
        return override
    if os.environ.get(ENV_VAR):
        return os.environ[ENV_VAR]
    return f"sqlite:///{_workspace_root() / 'market_cache.db'}"


def init_cache(engine: Engine) -> None:
    """Create the cache tables (only those) if they don't exist yet."""
    Base.metadata.create_all(engine, tables=list(CACHE_TABLES))
