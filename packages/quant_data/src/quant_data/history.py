import logging
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from quant_core.interfaces import BaseMarketDataProvider, CorporateActionProvider
from quant_core.models import CorporateAction, MarketDataRequest

logger = logging.getLogger(__name__)


def fetch_corporate_actions(
    provider: CorporateActionProvider, symbols: Iterable[str]
) -> dict[str, list[CorporateAction]]:
    """
    Full split and dividend history per symbol, raw amounts (PR 2a, D7).

    Sits beside ``fetch_daily_closes`` (which stays raw); callers combine the
    two with ``quant_core.analytics.adjust``. A symbol whose vendor call fails
    is logged and left out, not returned empty, so callers can tell "no
    actions" from "unknown".
    """
    actions = {}
    for symbol in sorted(set(symbols)):
        try:
            actions[symbol] = list(provider.get_corporate_actions(symbol))
        except Exception as e:
            logger.warning(f"No corporate actions for {symbol}: {e}")
    return actions


def fetch_daily_closes(
    provider: BaseMarketDataProvider,
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Daily closes, one column per symbol, indexed by UTC date.

    `end` is capped at today 00:00 UTC so a still-trading (partial) bar is never
    requested or cached. The index is normalised to the date because venues stamp
    daily bars at different times (BYMA 14:00 UTC, NYSE 13:30/14:30 UTC).
    Symbols that fail are logged and left out; callers detect them as missing columns.
    """
    end = min(
        end,
        datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
    )
    columns = {}
    for symbol in sorted(set(symbols)):
        request = MarketDataRequest(
            instrument_id=symbol,
            interval="1d",
            start_date=start,
            end_date=end,
            include_order_book=False,
            include_trades=False,
        )
        try:
            bars = provider.get_bars(request)
        except Exception as e:
            logger.warning(f"No daily history for {symbol}: {e}")
            continue
        index = pd.DatetimeIndex([b.timestamp for b in bars], tz="UTC").normalize()
        closes = pd.Series([b.close for b in bars], index=index)
        columns[symbol] = closes[~closes.index.duplicated(keep="last")]
    return pd.DataFrame(columns).sort_index()
