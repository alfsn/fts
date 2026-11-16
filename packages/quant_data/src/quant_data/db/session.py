from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool


def create_db_engine(database_url: str, **kwargs: Any) -> Engine:
    """
    Creates a SQLAlchemy engine configured for robust SQLite concurrency
    (WAL mode, NullPool to avoid stale pooled read transactions, 60s timeout).
    """
    connect_args = kwargs.pop("connect_args", {})
    if "sqlite" in database_url:
        connect_args.setdefault("timeout", 60)
        connect_args.setdefault("check_same_thread", False)
        if "poolclass" not in kwargs:
            in_memory = ":memory:" in database_url or database_url.endswith("://")
            kwargs["poolclass"] = StaticPool if in_memory else NullPool

    return create_engine(
        database_url,
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


def create_db_session(database_url: str) -> Session:
    """
    Creates a new database session bound to the specified database URL.
    """
    dyn_engine = create_db_engine(database_url)
    SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=dyn_engine)
    return SessionClass()
