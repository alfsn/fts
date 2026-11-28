"""The analyser's use cases (W2): one pipeline for every front end.

The CLI and the workbench call these functions, then print or draw what they
return. Nothing here prints or draws, and nothing here calculates: the maths
is `quant_core`, the mapping to it is `data_layer`.

    prepare_history  → cached raw closes + corporate actions
    build_snapshot   → today's holdings valued at the latest common close
    prepare_snapshot → both of the above
    prepare_review   → the snapshot, plus the ledger replayed over a date window,
                       the benchmark and the risk-free rate
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence, Union

import pandas as pd
from portfolio_analyser.data_layer.benchmark import (
    benchmark_label,
    benchmark_level,
    treasury_bill_rate,
)
from portfolio_analyser.data_layer.ledger_binding import (
    reconcile_snapshot,
    replay_ledger,
)
from portfolio_analyser.data_layer.market_history import (
    Actions,
    ccl_series,
    fx_context_at,
    history_symbols,
    history_window,
    price_positions,
    usd_per_unit,
    valuation_date,
)
from quant_core.analytics import window, window_bound
from quant_core.enums import Currency
from quant_core.ledger import Breaks
from quant_core.models import Benchmark, FXContext, Portfolio

Date = Union[datetime, pd.Timestamp, str, None]


class ReviewError(ValueError):
    """The inputs can't give a review (no ledger, no risk-free rate, ...).
    `breaks` is the reconciliation, when the run got that far, so the CLI can
    still show it."""

    def __init__(self, message: str, breaks: Optional[Breaks] = None):
        super().__init__(message)
        self.breaks = breaks


class QuantityBreakError(ReviewError):
    """Ledger and snapshot disagree on share counts: the run stops (D7)."""

    def __init__(self, breaks: Breaks):
        super().__init__(
            "Ledger and snapshot disagree on share counts: a trade or a split is missing.",
            breaks,
        )


@dataclass(frozen=True)
class HistoryRow:
    """Cache coverage of one symbol; `bars == 0` means no prices."""

    symbol: str
    first: Optional[pd.Timestamp]
    last: Optional[pd.Timestamp]
    bars: int


@dataclass(frozen=True)
class Snapshot:
    portfolio: Portfolio  # positions priced at `as_of`
    as_of: pd.Timestamp
    fx: pd.DataFrame  # USD per unit of each currency, per session
    fx_context: FXContext
    total_value: float
    breaks: Optional[Breaks]  # None without a ledger; quantity breaks raise


@dataclass(frozen=True)
class Review:
    snapshot: Snapshot
    values: pd.Series  # windowed ledger replay, base currency
    flows: pd.Series
    dropped: list  # unpriced days inside the window
    benchmark: Optional[pd.Series]  # total-return level, base currency
    benchmark_name: str
    risk_free: Union[float, pd.Series]  # NaN when there are no quotes
    warnings: list = field(default_factory=list)

    @property
    def period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.values.index[0], self.values.index[-1]


def prepare_history(
    portfolio: Portfolio,
    db_url: str | None,
    extra_symbols: Sequence[str] = (),
    sources=None,
) -> tuple[pd.DataFrame, Actions]:
    """Cache-backed raw daily closes and corporate actions for everything the
    snapshot and ledger need valuing, plus `extra_symbols` (benchmark,
    risk-free). `sources` are the providers behind the cache (default: Yahoo).
    Prices stay raw; `adjust` on read (PR 2a). `db_url` None means the shared
    cache (`quant_data.db.cache_url`, D18)."""
    from quant_data.db import cache_url, create_db_session, init_cache
    from quant_data.history import fetch_corporate_actions, fetch_daily_closes
    from quant_data.providers import TieredDataProvider, YFinanceMarketDataProvider

    symbols = history_symbols(portfolio) | set(extra_symbols)
    with create_db_session(cache_url(db_url)) as session:
        init_cache(session.get_bind())
        provider = TieredDataProvider(
            session, list(sources or [YFinanceMarketDataProvider()])
        )
        closes = fetch_daily_closes(provider, symbols, *history_window(portfolio))
        return closes, fetch_corporate_actions(provider, symbols)


def history_rows(portfolio: Portfolio, closes: pd.DataFrame) -> list[HistoryRow]:
    rows = []
    for symbol in sorted(history_symbols(portfolio)):
        series = closes[symbol].dropna() if symbol in closes else pd.Series(dtype=float)
        if series.empty:
            rows.append(HistoryRow(symbol, None, None, 0))
        else:
            rows.append(
                HistoryRow(symbol, series.index[0], series.index[-1], len(series))
            )
    return rows


def build_snapshot(
    portfolio: Portfolio, closes: pd.DataFrame, actions: Actions
) -> Snapshot:
    """Value the holdings at the latest session with every price and FX rate,
    and reconcile the ledger (if any) against them. Raises MissingFXRateError,
    ValueError (no price) or QuantityBreakError."""
    fx = usd_per_unit(ccl_series(closes, actions), closes.index)
    as_of = valuation_date(portfolio, closes, fx)
    fx_context = fx_context_at(portfolio, fx, as_of)
    price_positions(portfolio, closes, as_of)
    breaks = None
    if portfolio.transactions:
        breaks = reconcile_snapshot(portfolio, closes, actions, as_of)
        if not breaks.ok:
            raise QuantityBreakError(breaks)
    return Snapshot(
        portfolio=portfolio,
        as_of=as_of,
        fx=fx,
        fx_context=fx_context,
        total_value=portfolio.total_value(fx_context),
        breaks=breaks,
    )


def prepare_snapshot(
    portfolio: Portfolio, db_url: str | None, sources=None
) -> Snapshot:
    closes, actions = prepare_history(portfolio, db_url, sources=sources)
    return build_snapshot(portfolio, closes, actions)


def prepare_review(
    portfolio: Portfolio,
    db_url: str | None,
    *,
    benchmark_symbol: Optional[str] = None,
    risk_free_rate: Optional[float] = None,
    withholding: float = 0.0,
    start: Date = None,
    end: Date = None,
    sources=None,
) -> Review:
    """
    The ledger replayed day by day, valued at market, from the last close
    before `start` to `end` (default: the whole ledger up to the as-of date).
    KPIs rebase on that window; reconciliation is always at the as-of date.
    `risk_free_rate` is an annual fraction; None means the daily T-bill (^IRX),
    which only a USD portfolio may use.
    """
    from quant_data.reference import RISK_FREE_SYMBOL

    if not portfolio.transactions:
        raise ReviewError(
            "client-review needs a transactions ledger: performance is rebuilt "
            "day by day from it, never back-cast from today's holdings."
        )
    if risk_free_rate is None and portfolio.base_currency is not Currency.USD:
        raise ReviewError(
            f"Base currency {portfolio.base_currency.value}: the US T-bill is not "
            "its risk-free rate. Pass --risk-free-rate (annual fraction)."
        )
    benchmark = (
        portfolio.benchmark
        if benchmark_symbol is None
        else Benchmark(symbol=benchmark_symbol)
    )
    extra = [benchmark.symbol] + ([RISK_FREE_SYMBOL] if risk_free_rate is None else [])
    closes, actions = prepare_history(portfolio, db_url, extra, sources)
    snapshot = build_snapshot(portfolio, closes, actions)

    try:
        replayed = replay_ledger(
            portfolio, closes, actions, snapshot.fx, snapshot.as_of
        )
        warnings = _clip_warnings(replayed.values.index, start, end)
        values, flows = window(replayed.values, replayed.flows, start, end)
    except ValueError as e:
        raise ReviewError(str(e), snapshot.breaks) from e
    dropped = [d for d in replayed.dropped if values.index[0] <= d <= values.index[-1]]

    level = benchmark_level(
        closes,
        actions,
        benchmark,
        snapshot.fx,
        portfolio.base_currency,
        withholding,
    )
    if level is None:
        warnings.append(
            f"No prices for benchmark {benchmark.symbol}: "
            "benchmark statistics are n/a."
        )
    risk_free = risk_free_rate
    if risk_free is None:
        risk_free = treasury_bill_rate(closes)
        if risk_free is None:
            warnings.append(
                f"No {RISK_FREE_SYMBOL} quotes: Sharpe, Sortino, alpha "
                "and beta are n/a (or pass --risk-free-rate)."
            )
            risk_free = float("nan")
    return Review(
        snapshot=snapshot,
        values=values,
        flows=flows,
        dropped=dropped,
        benchmark=level,
        benchmark_name=benchmark_label(benchmark, withholding),
        risk_free=risk_free,
        warnings=warnings,
    )


def _clip_warnings(index: pd.DatetimeIndex, start: Date, end: Date) -> list[str]:
    """A window reaching outside the ledger is trimmed to it, with a warning (Q3)."""
    warnings = []
    if len(index) == 0:
        return warnings
    if start is not None and window_bound(start, index) < index[0]:
        warnings.append(
            f"Window starts at {index[0].date()}, the first valued ledger day."
        )
    if end is not None and window_bound(end, index, end=True) > index[-1]:
        warnings.append(f"Window ends at {index[-1].date()}, the valuation date.")
    return warnings
