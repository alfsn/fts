"""`uv run workbench`: start the Streamlit server on localhost only, since
reviews hold client data (Epic 8)."""

import argparse
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).with_name("app.py")


def command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the workbench.")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    try:
        subprocess.run(command(args.port), check=True)
    except KeyboardInterrupt:
        pass
