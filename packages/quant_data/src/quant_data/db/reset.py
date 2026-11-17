"""
Reset the market-data cache (PR 2a, D10).

Only the three cache tables are dropped and recreated. Everything else in the
same file (fts's model registry, Optuna studies, order/trade/prediction logs,
backtest equity) is untouched: it is not a cache.

    uv run python -m quant_data.db.reset --db-url sqlite:///apps/fts/dev.db
"""

import argparse

from sqlalchemy import func, inspect, select
from sqlalchemy.engine import Engine

from .models import BarDataLog, Base, CorporateActionRecord, FXRateRecord

CACHE_TABLES = (
    BarDataLog.__table__,
    CorporateActionRecord.__table__,
    FXRateRecord.__table__,
)


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
    parser.add_argument("--db-url", required=True, help="SQLAlchemy URL of the DB")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation")
    args = parser.parse_args(argv)

    from .session import create_db_engine

    engine = create_db_engine(args.db_url)
    counts = cache_row_counts(engine)
    print(
        f"{args.db_url}: dropping "
        + ", ".join(f"{t} ({n} rows)" for t, n in counts.items())
    )
    if not args.yes and input("Type 'yes' to continue: ").strip() != "yes":
        print("Aborted.")
        return 1
    reset_market_cache(engine)
    print("Done. Cache tables are empty; the next run repopulates them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
