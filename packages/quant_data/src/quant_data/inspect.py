"""
Read-only views of the shared price cache, for the workbench (W3b, D12).

Both apps share one cache, so the inspector lives here rather than in either
app. It only reads (``load_bars`` / ``load_corporate_actions``): it never
calls a vendor (import-linter contract: no ``quant_data.providers``). The
maths is quant_core's (``adjust``, ``find_gaps``, the CCL recipe); this
module maps cached rows to its inputs.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import pandas as pd
from quant_core.analytics import adjust
from quant_core.enums import PriceAdjustment
from quant_core.intervals import find_gaps
from quant_core.models import CorporateAction

from .db import (
    CachedSeries,
    MarketDataRepository,
    cache_url,
    create_db_session,
    init_cache,
)
from .history import daily_closes
from .reference import ARS_CCL_PAIRS, ars_ccl, ars_implied, gap_rule_for
from .reference.gaps import GAP_RULES

DAILY = "1d"


@dataclass(frozen=True)
class SeriesView:
    """One cached series, raw beside its adjusted views, with its holes."""

    instrument_id: str
    interval: str
    source: Optional[str]  # None when nothing is cached
    known_source: bool  # False: the gap rule is the default, not the vendor's
    raw: pd.Series  # closes as traded
    split: pd.Series  # raw / split factor
    total_return: pd.Series  # split-adjusted, dividends reinvested (rescales the past)
    actions: List[CorporateAction]
    max_gap: timedelta
    gaps: pd.DataFrame  # start, end, length


@dataclass(frozen=True)
class CCLView:
    """The ARS CCL (ARS per USD) as the analyser computes it, and each pair's."""

    ccl: pd.Series  # NaN where the pairs disagree by more than 5%
    implied: pd.DataFrame  # one column per pair ("GGAL.BA/GGAL")
    missing: List[str]  # pair symbols with no cached daily bars

    @property
    def rejected_days(self) -> int:
        """Days with two or more pairs priced whose rates disagreed (NaN CCL)."""
        priced = self.implied.notna().sum(axis=1)
        return int(((priced >= 2) & self.ccl.reindex(priced.index).isna()).sum())

    @property
    def single_pair_days(self) -> int:
        """Days priced by one pair only: its rate is used unchecked."""
        return int((self.implied.notna().sum(axis=1) == 1).sum())


def _repo(url: Optional[str]):
    session = create_db_session(cache_url(url))
    init_cache(session.get_bind())
    return session


def cached_series(url: Optional[str] = None) -> List[CachedSeries]:
    """Everything in the cache: one row per (symbol, interval, source)."""
    with _repo(url) as session:
        return MarketDataRepository(session).series_summary()


def inspect_series(
    instrument_id: str, interval: str, url: Optional[str] = None
) -> SeriesView:
    """
    The cached closes of one series as traded, split-adjusted and total-return
    adjusted, its corporate actions, and its gaps under the rule of the
    vendor that filled it. Raises ValueError if a dividend is not below the
    previous close (a data error, ``adjust``).
    """
    with _repo(url) as session:
        repo = MarketDataRepository(session)
        bars = repo.load_bars(instrument_id, interval)
        actions = repo.load_corporate_actions(instrument_id)
        source = repo.bar_source(instrument_id, interval)

    timestamps = [b.timestamp for b in bars]
    if interval == DAILY:
        raw = daily_closes(bars)
    else:
        raw = pd.Series(
            [b.close for b in bars],
            index=pd.DatetimeIndex(timestamps, tz="UTC"),
            dtype=float,
        )
    raw = raw.rename(instrument_id)
    max_gap = gap_rule_for(source or "").max_gap(interval)
    split, total = raw.copy(), raw.copy()
    if not raw.empty:
        split = adjust(raw, actions, PriceAdjustment.SPLIT)
        total = adjust(raw, actions, PriceAdjustment.TOTAL_RETURN)
    return SeriesView(
        instrument_id=instrument_id,
        interval=interval,
        source=source,
        known_source=source in GAP_RULES,
        raw=raw,
        split=split,
        total_return=total,
        actions=actions,
        max_gap=max_gap,
        gaps=find_gaps(timestamps, max_gap),
    )


def ccl_view(url: Optional[str] = None) -> CCLView:
    """The CCL from the cached daily closes of the reference pairs, the same
    recipe the analyser values ARS with (``quant_data.reference.ars_ccl``)."""
    symbols = sorted({s for local, adr, _ in ARS_CCL_PAIRS for s in (local, adr)})
    with _repo(url) as session:
        repo = MarketDataRepository(session)
        columns = {s: daily_closes(repo.load_bars(s, DAILY)) for s in symbols}
        actions = {s: repo.load_corporate_actions(s) for s in symbols}
    missing = [s for s, closes in columns.items() if closes.empty]
    closes = pd.DataFrame({s: c for s, c in columns.items() if not c.empty})
    closes = closes.sort_index()
    if closes.empty:
        return CCLView(pd.Series(dtype=float), pd.DataFrame(), missing)
    return CCLView(ars_ccl(closes, actions), ars_implied(closes, actions), missing)
