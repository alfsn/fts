# src/trading_bot/core/database.py

from typing import Any, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

# Import the settings object from your config file
# This assumes your config file defines a 'settings' instance
# of your Pydantic BaseSettings class.
from ..config import settings


def create_db_engine(database_url: str | None = None, **kwargs: Any) -> Engine:
    """
    Creates a SQLAlchemy engine configured for robust SQLite concurrency
    (WAL mode, NullPool to avoid stale pooled read transactions, 60s timeout).
    """
    url = database_url or settings.DATABASE_URL
    connect_args = kwargs.pop("connect_args", {})
    if "sqlite" in url:
        connect_args.setdefault("timeout", 60)
        connect_args.setdefault("check_same_thread", False)
        if "poolclass" not in kwargs:
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
            else:
                kwargs["poolclass"] = NullPool

    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args=connect_args,
        **kwargs,
    )


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    if type(dbapi_connection).__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        cursor.close()


# Create the default SQLAlchemy engine and SessionLocal factory
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Create a Base class for our ORM models to inherit from
class Base(DeclarativeBase):
    """
    The declarative base class for all SQLAlchemy ORM models.
    """

    pass


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

    target_engine = bind_engine if bind_engine is not None else engine
    Base.metadata.create_all(bind=target_engine)


def create_db_session(database_url: str) -> Session:
    """
    Creates a new database session bound to the specified database URL.
    """
    dyn_engine = create_db_engine(database_url)
    SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=dyn_engine)
    return SessionClass()


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
