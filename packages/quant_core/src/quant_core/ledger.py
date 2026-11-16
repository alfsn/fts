"""
Replay a transaction ledger into daily holdings, cash and market value (PR 2b).

Pure functions over pandas frames; the app supplies the calendar, raw closes,
corporate actions and FX. Conventions:

- Every event dated d is in the close-of-d value, and a flow is already in the
  value at its timestamp (matches ``analytics.period_returns``). A transaction
  dated on a non-session day takes effect on the next session.
- Only deposits, withdrawals and in-kind transfers are **external flows**.
  Dividends and fees are internal and part of the return.
- Valuation is raw close x share count; splits (from corporate actions, keyed
  by market symbol) multiply the quantity held at the previous close.
- A held listing or a needed FX rate without a value makes the day NaN, never
  0. NaN days are dropped before returns are computed, with a warning.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .enums import CorporateActionType, Currency, TransactionType
from .models import CorporateAction, Listing, Portfolio, Transaction, listing_symbols

logger = logging.getLogger(__name__)

# Sessions a price or FX rate may be carried over a local holiday before it is
# NaN rather than stale (D9). quant_data's MAX_CALENDAR_GAP (5 days) is the
# same tolerance on the fetching side.
MAX_STALE_SESSIONS = 5
QUANTITY_TOLERANCE = 1e-9
CASH_TOLERANCE = 0.005  # half a cent

_FLOW_SIGN = {
    TransactionType.DEPOSIT: 1.0,
    TransactionType.WITHDRAWAL: -1.0,
    TransactionType.TRANSFER_IN: 1.0,
    TransactionType.TRANSFER_OUT: -1.0,
}
_QUANTITY_SIGN = {
    TransactionType.BUY: 1.0,
    TransactionType.TRANSFER_IN: 1.0,
    TransactionType.SELL: -1.0,
    TransactionType.TRANSFER_OUT: -1.0,
}


# --- calendar helpers ----------------------------------------------------------------


def _session_index(dates: pd.DatetimeIndex, day: date, what: str) -> int:
    """Position of the first session on or after ``day``."""
    stamp = pd.Timestamp(day)
    if dates.tz is not None:
        stamp = stamp.tz_localize(dates.tz)
    position = int(dates.searchsorted(stamp))
    if position >= len(dates):
        raise ValueError(
            f"{what} dated {day} is after the last valued day {dates[-1].date()}."
        )
    return position


def _action_day(action: CorporateAction) -> date:
    stamp = pd.Timestamp(action.ex_date)
    return (stamp.tz_localize(None) if stamp.tz is not None else stamp).date()


def _positions(
    transactions: Sequence[Transaction], dates: pd.DatetimeIndex
) -> List[int]:
    """Session position of every transaction; any line past the calendar raises."""
    return [_session_index(dates, t.date, "Transaction") for t in transactions]


def symbols_by_listing(transactions: Iterable[Transaction]) -> Dict[Listing, str]:
    return listing_symbols(transactions)


def forward_fill(
    frame: pd.DataFrame, dates: pd.DatetimeIndex, limit: int = MAX_STALE_SESSIONS
) -> pd.DataFrame:
    """Align to the calendar and carry the last print over at most ``limit``
    sessions. Only past values are used, so there is no look-ahead."""
    return frame.reindex(dates).ffill(limit=limit)


# --- replay ----------------------------------------------------------------------------


def holdings(
    transactions: Sequence[Transaction],
    actions: Mapping[str, Sequence[CorporateAction]],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Share count per listing (columns ``(instrument_id, currency)``) per session.
    On a split's ex-date the quantity held at the previous close is multiplied
    by the ratio, then that day's trades are added (they are in new shares).
    """
    symbols = symbols_by_listing(transactions)
    deltas: Dict[Listing, np.ndarray] = {k: np.zeros(len(dates)) for k in symbols}
    for t, position in zip(transactions, _positions(transactions, dates)):
        if t.type in _QUANTITY_SIGN:
            deltas[t.listing][position] += _QUANTITY_SIGN[t.type] * t.quantity

    columns = {}
    for listing, delta in deltas.items():
        ratios: Dict[int, float] = {}
        for action in actions.get(symbols[listing], ()):
            if action.action_type is not CorporateActionType.SPLIT:
                continue
            day = _action_day(action)
            if dates[0].date() < day <= dates[-1].date():
                position = _session_index(dates, day, "Split")
                ratios[position] = ratios.get(position, 1.0) * action.ratio
        quantity, series = 0.0, np.zeros(len(dates))
        for i in range(len(dates)):
            quantity = quantity * ratios.get(i, 1.0) + delta[i]
            series[i] = quantity
        columns[listing] = series
    frame = pd.DataFrame(columns, index=dates)
    frame.columns = pd.MultiIndex.from_tuples(
        list(columns), names=["instrument_id", "currency"]
    )
    return frame


def cash(transactions: Sequence[Transaction], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Cash per currency per session. Transfers never touch cash."""
    currencies = sorted({t.currency for t in transactions}, key=lambda c: c.value)
    deltas = {c: np.zeros(len(dates)) for c in currencies}
    for t, position in zip(transactions, _positions(transactions, dates)):
        if t.type is TransactionType.DEPOSIT or t.type is TransactionType.DIVIDEND:
            change = t.amount
        elif t.type is TransactionType.WITHDRAWAL or t.type is TransactionType.FEE:
            change = -t.amount
        elif t.type is TransactionType.BUY:
            change = -(t.quantity * t.price + t.fees)
        elif t.type is TransactionType.SELL:
            change = t.quantity * t.price - t.fees
        else:
            continue
        deltas[t.currency][position] += change
    return pd.DataFrame(deltas, index=dates).cumsum()


def external_flows(
    transactions: Sequence[Transaction],
    closes: pd.DataFrame,
    fx: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Deposits (+), withdrawals (−) and in-kind transfers (± quantity x raw close
    on that day) in base currency, one entry per session (0 when none). These
    are the ``cash_flows`` for ``compute_performance_metrics``. ``closes`` and
    ``fx`` are already forward-filled; a transfer without a price, or a flow
    without an FX rate, raises: a flow needs a valuation.
    """
    symbols = symbols_by_listing(transactions)
    flows = pd.Series(0.0, index=dates)
    for t, position in zip(transactions, _positions(transactions, dates)):
        sign = _FLOW_SIGN.get(t.type)
        if sign is None:
            continue
        day = dates[position]
        rate = _at(fx, t.currency, day)
        if not rate > 0:
            raise ValueError(f"No FX rate for {t.currency.value} on {day.date()}.")
        if t.type in (TransactionType.DEPOSIT, TransactionType.WITHDRAWAL):
            local = t.amount
        else:
            symbol = symbols[t.listing]
            price = _at(closes, symbol, day)
            if not price > 0:
                raise ValueError(
                    f"{symbol} has no price on {day.date()} to value the {t.type.value}."
                )
            local = t.quantity * price
        flows.loc[day] += sign * local * rate
    return flows


def _at(frame: pd.DataFrame, column, day) -> float:
    if column not in frame.columns:
        return float("nan")
    return float(frame.at[day, column])


def portfolio_values(
    holdings: pd.DataFrame,
    cash: pd.DataFrame,
    closes: pd.DataFrame,
    fx: pd.DataFrame,
    symbols: Mapping[Listing, str],
) -> pd.Series:
    """
    Σ quantity x raw close x FX + Σ cash x FX, in base currency, per session.
    NaN wherever a held (non-zero) listing has no close or a non-zero balance
    has no FX rate. A listing held at zero never matters.
    """
    dates = holdings.index
    value = pd.Series(0.0, index=dates)
    for listing in holdings.columns:
        quantity = holdings[listing]
        price = _column(closes, symbols[listing], dates)
        rate = _column(fx, listing[1], dates)
        value = value + (quantity * price * rate).where(quantity != 0, 0.0)
    for currency in cash.columns:
        balance = cash[currency]
        rate = _column(fx, currency, dates)
        value = value + (balance * rate).where(balance != 0, 0.0)
    return value


def _column(frame: pd.DataFrame, column, dates: pd.DatetimeIndex) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=dates)
    return frame[column].reindex(dates)


def drop_unvalued_days(values: pd.Series) -> pd.Series:
    """Drop NaN days with a warning naming them (D10). Dropping them *before*
    returns are taken keeps the return across the hole; leaving them in would
    silently lose it (pandas ``cumprod`` skips NaN)."""
    missing = values.index[values.isna()]
    if len(missing):
        days = ", ".join(str(d.date()) for d in missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        logger.warning(
            f"{len(missing)} day(s) without a valuation dropped: {days}{more}"
        )
    return values.dropna()


@dataclass
class ReplayResult:
    holdings: pd.DataFrame
    cash: pd.DataFrame
    values: pd.Series  # NaN days already dropped
    flows: pd.Series  # non-zero flows kept even on dropped days, so they raise
    dropped: List[pd.Timestamp] = field(default_factory=list)


def replay(
    transactions: Sequence[Transaction],
    closes: pd.DataFrame,
    fx: pd.DataFrame,
    actions: Mapping[str, Sequence[CorporateAction]],
    dates: pd.DatetimeIndex,
) -> ReplayResult:
    """The whole pipeline: forward-fill prices and FX, replay, value, drop NaN days."""
    closes, fx = forward_fill(closes, dates), forward_fill(fx, dates)
    held, balances = holdings(transactions, actions, dates), cash(transactions, dates)
    values = portfolio_values(
        held, balances, closes, fx, symbols_by_listing(transactions)
    )
    flows = external_flows(transactions, closes, fx, dates)
    kept = drop_unvalued_days(values)
    dropped = list(values.index.difference(kept.index))
    flows = flows[flows.index.isin(kept.index) | (flows != 0)]
    return ReplayResult(held, balances, kept, flows, dropped)


# --- reconciliation (D4, D7) -----------------------------------------------------------


@dataclass
class Breaks:
    """Ledger vs snapshot on the snapshot date: ``(ledger, snapshot)`` per key."""

    quantity: Dict[Listing, Tuple[float, float]] = field(default_factory=dict)
    cash: Dict[Currency, Tuple[float, float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """A quantity break is an error; a cash break is only a warning (D7)."""
        return not self.quantity

    def describe(self) -> str:
        lines = [
            f"quantity break {iid} ({ccy.value}): ledger {ledger:g}, snapshot {snap:g}"
            for (iid, ccy), (ledger, snap) in self.quantity.items()
        ] + [
            f"cash break {ccy.value}: ledger {ledger:,.2f}, snapshot {snap:,.2f} "
            f"(unexplained {snap - ledger:,.2f})"
            for ccy, (ledger, snap) in self.cash.items()
        ]
        return "\n".join(lines)


def reconcile(
    holdings_at: pd.Series, cash_at: pd.Series, portfolio: Portfolio
) -> Breaks:
    """Compare the ledger's holdings and cash on one day with the snapshot."""
    ledger_quantity: Dict[Listing, float] = {
        (str(iid), Currency(ccy)): float(q)  # pandas stores the enum level as str
        for (iid, ccy), q in holdings_at.items()
        if q != 0
    }
    snapshot_quantity: Dict[Listing, float] = {}
    for p in portfolio.positions:
        key = (p.instrument_id, p.currency)
        snapshot_quantity[key] = snapshot_quantity.get(key, 0.0) + p.quantity
    ledger_cash = {Currency(c): float(v) for c, v in cash_at.items()}
    snapshot_cash: Dict[Currency, float] = {}
    for c in portfolio.cash_balances:
        snapshot_cash[c.currency] = snapshot_cash.get(c.currency, 0.0) + c.total

    breaks = Breaks()
    for key in sorted(set(ledger_quantity) | set(snapshot_quantity), key=str):
        ledger, snap = ledger_quantity.get(key, 0.0), snapshot_quantity.get(key, 0.0)
        if abs(ledger - snap) > QUANTITY_TOLERANCE:
            breaks.quantity[key] = (ledger, snap)
    for currency in sorted(
        set(ledger_cash) | set(snapshot_cash), key=lambda c: c.value
    ):
        ledger, snap = ledger_cash.get(currency, 0.0), snapshot_cash.get(currency, 0.0)
        if abs(ledger - snap) > CASH_TOLERANCE:
            breaks.cash[currency] = (ledger, snap)
    return breaks
