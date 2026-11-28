"""The workbench's starting database URLs: whatever the environment says.
Blank means "the app's own default" (fts settings, ``quant_data.db.cache_url``);
the workbench never holds a default path itself."""

import os


def research_url() -> str:
    """fts's research DB (models, runs): ``DATABASE_URL``."""
    return os.environ.get("DATABASE_URL", "")


def market_cache_url() -> str:
    """The shared price cache: ``MARKET_CACHE_URL``."""
    return os.environ.get("MARKET_CACHE_URL", "")
