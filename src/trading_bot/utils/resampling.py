# src/trading_bot/utils/resampling.py

import logging
from datetime import timezone
from typing import List, Sequence

import pandas as pd

from trading_bot.core.enums import BarType
from trading_bot.core.schemas import BarData

logger = logging.getLogger(__name__)

INTERVAL_TO_PANDAS = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
}


def timeframe_to_seconds(tf: str) -> int:
    """Converts a timeframe string (e.g. '1m', '2h', '1d') to equivalent seconds."""
    if not tf:
        raise ValueError("Timeframe string cannot be empty.")

    unit = tf[-1].lower()
    value_str = tf[:-1]

    if not value_str.isdigit():
        raise ValueError(f"Invalid timeframe value format: {tf}")

    value = int(value_str)

    if unit == "m":
        return value * 60
    elif unit == "h":
        return value * 3600
    elif unit == "d":
        return value * 86400
    elif unit == "w":
        return value * 604800
    else:
        raise ValueError(f"Unknown timeframe unit '{unit}' in {tf}")


def resample_bars(bars: Sequence[BarData], target_interval: str) -> List[BarData]:
    """
    Downsamples a sequence of BarData models to a lower target frequency (e.g. 1m to 1h).

    :param bars: Sequence of higher frequency BarData records.
    :param target_interval: Target timeframe string (e.g. '5m', '1h', '1d').
    :return: A list of resampled BarData records.
    """
    if not bars:
        return []

    # Detect source interval from first bar
    source_interval = bars[0].interval
    if not source_interval:
        # Fallback: calculate interval from timestamps
        if len(bars) > 1:
            diff_seconds = int((bars[1].timestamp - bars[0].timestamp).total_seconds())
            if diff_seconds % 604800 == 0:
                source_interval = f"{diff_seconds // 604800}w"
            elif diff_seconds % 86400 == 0:
                source_interval = f"{diff_seconds // 86400}d"
            elif diff_seconds % 3600 == 0:
                source_interval = f"{diff_seconds // 3600}h"
            else:
                source_interval = f"{diff_seconds // 60}m"
        else:
            source_interval = "1m"

    # Validate interval durations (cannot upsample)
    try:
        src_dur = timeframe_to_seconds(source_interval)
        tgt_dur = timeframe_to_seconds(target_interval)
        if tgt_dur < src_dur:
            raise ValueError(
                f"Target interval {target_interval} cannot be a higher frequency "
                f"than source interval {source_interval}."
            )
    except Exception as e:
        raise ValueError(f"Failed to validate intervals for downsampling: {e}")

    # Build Pandas DataFrame
    data = []
    for bar in bars:
        data.append(
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "ticks_count": bar.ticks_count,
                "dollar_volume": bar.dollar_volume,
            }
        )
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)

    # Map target_interval to Pandas offset
    pandas_offset = INTERVAL_TO_PANDAS.get(target_interval.lower())
    if not pandas_offset:
        unit = target_interval[-1].lower()
        val = target_interval[:-1]
        if unit == "m":
            pandas_offset = f"{val}min"
        else:
            pandas_offset = target_interval

    # Resample using standard OHLCV rules: closed="left", label="left"
    resampled_df = df.resample(pandas_offset, closed="left", label="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "ticks_count": "sum",
            "dollar_volume": "sum",
        }
    )

    # Drop intervals with no trade activity (empty rows)
    resampled_df.dropna(subset=["close"], inplace=True)

    # Convert back to BarData schema list
    resampled_bars = []
    bar_type = bars[0].bar_type
    for ts, row in resampled_df.iterrows():
        # Make timestamp timezone-aware (UTC) if naive
        dt = ts.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        resampled_bars.append(
            BarData(
                timestamp=dt,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                bar_type=bar_type,
                interval=target_interval,
                ticks_count=int(row["ticks_count"]),
                dollar_volume=float(row["dollar_volume"]),
            )
        )

    logger.debug(
        f"Resampled {len(bars)} source bars ({source_interval}) "
        f"into {len(resampled_bars)} target bars ({target_interval})."
    )
    return resampled_bars
