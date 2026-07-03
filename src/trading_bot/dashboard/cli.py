# src/trading_bot/dashboard/cli.py

"""
CLI Launcher for Quant Data Catalog & Streamlit Dashboard.
Usage: python -m trading_bot.dashboard.cli --db dev.db --port 8501
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Quant Data Catalog & Backtest Explorer Streamlit Dashboard."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="dev.db",
        help="Database file path or connection URL (e.g. dev.db or sqlite:///dev.db or postgresql://...)",
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

    # Format database URL
    db_url = args.db
    if not (
        db_url.startswith("sqlite:")
        or db_url.startswith("postgresql:")
        or db_url.startswith("mysql:")
    ):
        db_url = f"sqlite:///{db_url}"

    os.environ["DATABASE_URL"] = db_url

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

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard server stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit dashboard: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
