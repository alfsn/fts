# src/trading_bot/dashboard/cli.py

"""
CLI Launcher for Quant Data Catalog & Streamlit Dashboard.
Usage: python -m trading_bot.dashboard.cli --db dev.db --cache path/to/cache.db --port 8501
"""

import argparse
import os
import subprocess
import sys

from quant_data.db import cache_url
from trading_bot.config import get_settings


def _as_url(path_or_url: str) -> str:
    """A bare file path becomes a SQLite URL; a URL (any dialect and driver,
    e.g. ``sqlite+pysqlite:///x.db``) is kept as given."""
    if "://" in path_or_url:
        return path_or_url
    return f"sqlite:///{path_or_url}"


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Quant Data Catalog & Backtest Explorer Streamlit Dashboard."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database file path or connection URL (defaults to settings.DATABASE_URL)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Price-cache file path or URL (defaults to MARKET_CACHE_URL, else the shared workspace cache)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the Streamlit dashboard server on (default: 8501)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    settings = get_settings()
    db_url = _as_url(args.db if args.db is not None else settings.DATABASE_URL)
    market_cache = cache_url(
        _as_url(args.cache) if args.cache is not None else settings.MARKET_CACHE_URL
    )
    # The Streamlit subprocess reads both from its environment.
    os.environ["DATABASE_URL"] = db_url
    os.environ["MARKET_CACHE_URL"] = market_cache

    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]

    print(
        f"⚡ Launching Quant Data Catalog Dashboard on http://{args.host}:{args.port}..."
    )
    print(f"📂 Connected Database: {db_url}")
    print(f"📂 Price cache: {market_cache}")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard server stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit dashboard: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
