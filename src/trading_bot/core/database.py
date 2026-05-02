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


def init_db() -> None:
    """
    A utility function to create all tables in the database.
    This should be called once on application startup.
    """
    Base.metadata.create_all(bind=engine)


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
