# src/trading_bot/runs/integrity.py
"""
Fills are the truth, positions are derived (D19).

At the end of a backtest the saved fills are replayed with the ``quant_core``
ledger and must give the Portfolio's final positions and cash. A missing,
duplicated or wrongly priced fill row fails the run instead of leaving a
quietly wrong record behind.
"""

from typing import Dict, List, Mapping

import pandas as pd
from quant_core import ledger
from quant_core.enums import Currency, TransactionType
from quant_core.models import Transaction

from ..core.enums import OrderSide
from .records import RunRecord


class RunIntegrityError(RuntimeError):
    """The saved fills don't replay to the Portfolio's final state."""


def position_key(instrument_id: str, outcome) -> str:
    """The Portfolio's key: YES and NO of one market are separate books."""
    return f"{instrument_id}:{outcome}" if outcome else instrument_id


def to_transactions(record: RunRecord, currency: str) -> List[Transaction]:
    """The run as ledger lines: the initial cash as one deposit, then one
    buy or sell per fill (keyed like the Portfolio, so outcomes never merge)."""
    ccy = Currency(currency)  # an unknown quote currency raises here
    if record.header.initial_cash is None:
        raise RunIntegrityError(f"Run {record.header.run_id} has no initial cash.")
    first = min(
        [f.fill_timestamp for f in record.fills]
        + [p.timestamp for p in record.equity]
        + ([record.header.start_time] if record.header.start_time else []),
        default=None,
    )
    lines: List[Transaction] = []
    if record.header.initial_cash > 0:
        lines.append(
            Transaction(
                date=first.date(),
                type=TransactionType.DEPOSIT,
                currency=ccy,
                amount=record.header.initial_cash,
            )
        )
    for f in record.fills:
        lines.append(
            Transaction(
                date=f.fill_timestamp.date(),
                type=(
                    TransactionType.BUY
                    if f.side == OrderSide.BUY
                    else TransactionType.SELL
                ),
                currency=ccy,
                instrument_id=position_key(f.instrument_id, f.outcome),
                quantity=f.quantity,
                price=f.fill_price,
            )
        )
    return lines


def check_fills_match(
    record: RunRecord,
    positions: Mapping[str, float],
    cash: float,
    currency: str,
) -> None:
    """Raise ``RunIntegrityError`` when replaying the fills doesn't give
    ``positions`` (Portfolio key -> quantity) and ``cash``."""
    lines = to_transactions(record, currency)
    if not lines:
        return
    dates = pd.DatetimeIndex(sorted({pd.Timestamp(t.date) for t in lines}))
    held = ledger.holdings(lines, {}, dates).iloc[-1]
    replayed: Dict[str, float] = {
        str(iid): float(q) for (iid, _), q in held.items() if q != 0
    }
    balances = ledger.cash(lines, dates).iloc[-1]
    replayed_cash = float(balances.sum())

    problems = []
    for key in sorted(set(replayed) | set(positions)):
        a, b = replayed.get(key, 0.0), positions.get(key, 0.0)
        if abs(a - b) > ledger.QUANTITY_TOLERANCE:
            problems.append(f"{key}: fills give {a}, portfolio holds {b}")
    if abs(replayed_cash - cash) > ledger.CASH_TOLERANCE:
        problems.append(f"cash: fills give {replayed_cash}, portfolio has {cash}")
    if problems:
        raise RunIntegrityError(
            f"Run {record.header.run_id}: saved fills don't match the portfolio. "
            + "; ".join(problems)
        )
