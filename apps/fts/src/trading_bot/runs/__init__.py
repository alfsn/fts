# src/trading_bot/runs/__init__.py
"""
Runs (M1): the one package that knows which tables make up a run.

Write during a run with a ``RunRecorder``; read, list or delete whole runs
with ``RunStore``. Nothing outside this package imports ``runs.tables``.
"""

from .integrity import RunIntegrityError, check_fills_match, position_key
from .recorder import RunRecorder, SqlRunRecorder
from .records import (
    EquityPoint,
    Fill,
    Prediction,
    RunHeader,
    RunKind,
    RunRecord,
    Segment,
)
from .store import RunStore, delete_run, load_record

__all__ = [
    "EquityPoint",
    "Fill",
    "Prediction",
    "RunHeader",
    "RunIntegrityError",
    "RunKind",
    "RunRecord",
    "RunRecorder",
    "RunStore",
    "Segment",
    "SqlRunRecorder",
    "check_fills_match",
    "delete_run",
    "load_record",
    "position_key",
]
