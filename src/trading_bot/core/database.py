# src/trading_bot/core/database.py

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Import the settings object from your config file
# This assumes your config file defines a 'settings' instance
# of your Pydantic BaseSettings class.
from ..config import settings

# Create the SQLAlchemy engine using the URL from settings
engine = create_engine(
    settings.DATABASE_URL,
    # pool_pre_ping=True ensures the connection is
    # valid before being used from the pool.
    pool_pre_ping=True,
)

# Create a configured "Session" class
# This is the factory that will create new session objects
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Create a Base class for our ORM models to inherit from
# This replaces the old declarative_base() function
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
    dyn_engine = create_engine(database_url, pool_pre_ping=True)
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
