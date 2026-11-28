# src/trading_bot/core/database.py

from typing import Any, Generator

from quant_data.db.session import create_db_engine as _create_db_engine
from quant_data.db.session import create_db_session  # noqa: F401  (re-export)
from sqlalchemy import create_engine  # noqa: F401  (re-export)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Import the settings object from your config file
# This assumes your config file defines a 'settings' instance
# of your Pydantic BaseSettings class.
from ..config import settings


def create_db_engine(database_url: str | None = None, **kwargs: Any) -> Engine:
    """Shared quant_data engine factory, defaulting to fts settings."""
    return _create_db_engine(database_url or settings.DATABASE_URL, **kwargs)


# Create the default SQLAlchemy engine and SessionLocal factory
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Create a Base class for our ORM models to inherit from
from quant_data.db import Base


def init_db(extra_models: list[str] | None = None, bind_engine=None) -> None:
    """
    A utility function to create all tables in the database.
    This should be called once on application startup.

    :param extra_models: A list of module strings
    (e.g., ['plugins.forecasting.db_models'])
                         to import before creating tables, ensuring plugin models
                         are registered with Base.metadata.
    :param bind_engine: Optional SQLAlchemy Engine to bind to. If None, default engine is used.
    """
    if extra_models:
        import importlib

        for module_name in extra_models:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                # We could log this or re-raise. For now, we'll re-raise
                # as missing models are a critical failure for DB init.
                raise ImportError(
                    f"Failed to import extra models from {module_name}: {e}"
                ) from e

    # Register every fts table with Base.metadata (imported here: they import Base).
    from .. import runs  # noqa: F401
    from . import models  # noqa: F401

    target_engine = bind_engine if bind_engine is not None else engine
    Base.metadata.create_all(bind=target_engine)


def get_db() -> Generator[Session, None, None]:
    """
    A generator function that acts as a session context manager.

    This is a best practice for managing session lifecycles,
    ensuring that sessions are always closed, even if errors occur.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
