# src/trading_bot/core/database.py
"""
fts's two databases (D16, D18):

- research (``DATABASE_URL``): model registry, datasets, runs, Optuna. Its
  tables inherit from ``ResearchBase``; ``SessionLocal`` opens it.
- price cache (``MARKET_CACHE_URL``, shared with the analyser): bars,
  corporate actions, FX. Only ``quant_data`` knows its tables;
  ``CacheSessionLocal`` opens it.
"""

from typing import Any, Generator

from quant_data.db import cache_url, init_cache
from quant_data.db.session import create_db_engine as _create_db_engine
from quant_data.db.session import create_db_session  # noqa: F401  (re-export)
from sqlalchemy import create_engine  # noqa: F401  (re-export)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..config import settings


class ResearchBase(DeclarativeBase):
    """Registry of fts's research tables (never the price cache)."""


def create_db_engine(database_url: str | None = None, **kwargs: Any) -> Engine:
    """Shared quant_data engine factory, defaulting to fts's research DB."""
    return _create_db_engine(database_url or settings.DATABASE_URL, **kwargs)


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

cache_engine = create_db_engine(cache_url())
CacheSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cache_engine)


def init_db(extra_models: list[str] | None = None, bind_engine=None) -> None:
    """
    Create fts's research tables (never the cache's).

    :param extra_models: Module names to import first, so plugin models are
        registered with ``ResearchBase.metadata``.
    :param bind_engine: Engine to bind to; defaults to the research engine.
    """
    if extra_models:
        import importlib

        for module_name in extra_models:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import extra models from {module_name}: {e}"
                ) from e

    # Register every fts table with ResearchBase.metadata (they import it).
    from .. import runs  # noqa: F401
    from . import models  # noqa: F401

    target_engine = bind_engine if bind_engine is not None else engine
    ResearchBase.metadata.create_all(bind=target_engine)


def init_market_cache() -> None:
    """Create the price-cache tables in the cache file if they're missing."""
    init_cache(CacheSessionLocal.kw["bind"])


def get_db() -> Generator[Session, None, None]:
    """Yield a research session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
