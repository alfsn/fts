"""
Reset the shared market-data cache (D10, D18).

Only the cache tables (``markets``, ``bar_data_logs``, ``corporate_actions``,
``fx_rates``) are dropped and recreated; anything else in the same file is
untouched. The cache is ``market_cache.db`` in the workspace root unless
``MARKET_CACHE_URL`` or ``--db-url`` says otherwise.

    uv run python -m quant_data.db.reset
"""

import argparse

from sqlalchemy import func, inspect, select
from sqlalchemy.engine import Engine

from .cache import CACHE_TABLES, cache_url
from .models import Base


def cache_row_counts(engine: Engine) -> dict[str, int]:
    """Rows per cache table; 0 for a table that doesn't exist yet."""
    existing = set(inspect(engine).get_table_names())
    counts = {}
    with engine.connect() as conn:
        for table in CACHE_TABLES:
            counts[table.name] = (
                conn.execute(select(func.count()).select_from(table)).scalar_one()
                if table.name in existing
                else 0
            )
    return counts


def reset_market_cache(engine: Engine) -> None:
    """Drop and recreate exactly ``CACHE_TABLES`` with the current schema."""
    Base.metadata.drop_all(engine, tables=list(CACHE_TABLES), checkfirst=True)
    Base.metadata.create_all(engine, tables=list(CACHE_TABLES))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--db-url",
        default=None,
        help="SQLAlchemy URL of the cache (default: $MARKET_CACHE_URL, else market_cache.db)",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation")
    args = parser.parse_args(argv)

    from .session import create_db_engine

    db_url = cache_url(args.db_url)
    engine = create_db_engine(db_url)
    counts = cache_row_counts(engine)
    print(
        f"{db_url}: dropping " + ", ".join(f"{t} ({n} rows)" for t, n in counts.items())
    )
    if not args.yes and input("Type 'yes' to continue: ").strip() != "yes":
        print("Aborted.")
        return 1
    reset_market_cache(engine)
    print("Done. Cache tables are empty; the next run repopulates them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
