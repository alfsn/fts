"""
Reset the run tables (M1, D15: old runs are disposable).

Only the run tables (and the retired ``positions`` table) are dropped and
recreated. The model registry, datasets and Optuna studies in the same file are
untouched; the price cache lives in its own file (``MARKET_CACHE_URL``, D18).

    uv run python -m trading_bot.runs.reset --db-url sqlite:///apps/fts/dev.db
"""

import argparse

from sqlalchemy import MetaData, Table, func, inspect, select
from sqlalchemy.engine import Engine

from ..core.database import ResearchBase
from .tables import RUN_TABLES

# Positions are derived from fills since M1 (D19); the table is dropped, not recreated.
RETIRED_TABLES = ("positions",)


def run_row_counts(engine: Engine) -> dict[str, int]:
    """Rows per run table (and retired table) that exists."""
    existing = set(inspect(engine).get_table_names())
    names = [t.name for t in RUN_TABLES] + list(RETIRED_TABLES)
    counts = {}
    with engine.connect() as conn:
        for name in names:
            if name in existing:
                table = Table(name, MetaData(), autoload_with=engine, resolve_fks=False)
                counts[name] = conn.execute(
                    select(func.count()).select_from(table)
                ).scalar_one()
    return counts


def reset_runs(engine: Engine) -> None:
    """Drop every run table and the retired ones, then recreate the run tables."""
    existing = set(inspect(engine).get_table_names())
    for name in RETIRED_TABLES:
        if name in existing:
            # resolve_fks=False: reflecting must not pull in (and drop) the
            # tables it points at, e.g. ``markets``.
            Table(name, MetaData(), autoload_with=engine, resolve_fks=False).drop(
                engine
            )
    ResearchBase.metadata.drop_all(engine, tables=list(RUN_TABLES), checkfirst=True)
    ResearchBase.metadata.create_all(engine, tables=list(RUN_TABLES))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db-url", required=True, help="SQLAlchemy URL of the DB")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation")
    args = parser.parse_args(argv)

    from quant_data.db.session import create_db_engine

    engine = create_db_engine(args.db_url)
    counts = run_row_counts(engine)
    print(
        f"{args.db_url}: dropping "
        + (", ".join(f"{t} ({n} rows)" for t, n in counts.items()) or "nothing")
    )
    if not args.yes and input("Type 'yes' to continue: ").strip() != "yes":
        print("Aborted.")
        return 1
    reset_runs(engine)
    print("Done. Run tables are empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
