"""Drawdown analytics (PR 5). Feed a ``wealth_index`` when the series has cash flows."""

from typing import List

import pandas as pd

from ..models import DrawdownEpisode


def drawdown_series(values: pd.Series) -> pd.Series:
    """Fractional distance below the running high-water mark (<= 0)."""
    return values / values.cummax() - 1


def compute_max_drawdown(values: pd.Series) -> float:
    return float(drawdown_series(values).min()) if len(values) else float("nan")


def drawdown_episodes(values: pd.Series) -> List[DrawdownEpisode]:
    """All peak -> trough -> recovery episodes, in chronological order."""
    dd = drawdown_series(values)
    # Every new high starts a group; a group longer than one row is an episode.
    group_ids = (dd >= 0).cumsum()
    episodes = []
    for _, group in dd.groupby(group_ids):
        if len(group) < 2:
            continue
        peak = group.index[0]
        end_pos = values.index.get_loc(group.index[-1]) + 1
        recovery = values.index[end_pos] if end_pos < len(values) else None
        episodes.append(
            DrawdownEpisode(
                peak=peak,
                trough=group.idxmin(),
                recovery=recovery,
                depth=float(group.min()),
                duration_days=((recovery or group.index[-1]) - peak).total_seconds()
                / 86400,
            )
        )
    return episodes


def top_drawdowns(values: pd.Series, n: int = 5) -> List[DrawdownEpisode]:
    return sorted(drawdown_episodes(values), key=lambda e: e.depth)[:n]


def max_time_underwater_days(values: pd.Series) -> float:
    episodes = drawdown_episodes(values)
    return max((e.duration_days for e in episodes), default=0.0)
