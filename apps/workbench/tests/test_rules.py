"""The workbench's own rules that aren't imports (W3a)."""

import math
import re
from pathlib import Path

from workbench.cli import command
from workbench.filters import keep, numeric_fields

SRC = Path(__file__).resolve().parents[1] / "src" / "workbench"
# Maths belongs to quant_core (AGENTS.md "UI code never calculates").
MATHS = re.compile(r"\.cummax\(|\.pct_change\(|\.cumprod\(|\*\s*100\b")


def test_no_maths_in_the_workbench():
    hits = [
        f"{p.name}:{n}"
        for p in SRC.rglob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if MATHS.search(line)
    ]
    assert hits == []


def test_a_nan_parameter_fails_a_range_and_is_not_a_bound():
    params = [{"lr": 0.1}, {"lr": math.nan}, {"lr": 0.3}]
    assert numeric_fields(params) == {"lr": (0.1, 0.3)}
    assert [keep(p, {"lr": (0.0, 1.0)}, "") for p in params] == [True, False, True]


def test_the_server_listens_on_localhost_only():
    cmd = command(8501)
    assert cmd[cmd.index("--server.address") + 1] == "127.0.0.1"
