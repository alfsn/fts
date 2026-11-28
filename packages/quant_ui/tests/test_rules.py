"""quant_ui only draws (AGENTS.md "UI code never calculates", W4)."""

import re
from pathlib import Path

import quant_ui

SRC = Path(quant_ui.__file__).resolve().parent
# Return and drawdown maths belong to quant_core. `* 100` is allowed here:
# turning a stored fraction into a percent is the UI's job.
MATHS = re.compile(r"\.cummax\(|\.pct_change\(|\.cumprod\(")


def test_no_maths_in_quant_ui():
    hits = [
        f"{p.name}:{n}"
        for p in SRC.rglob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if MATHS.search(line)
    ]
    assert hits == []
