"""Trade-level statistics. A fill on its own is neither a win nor a loss: only
closed round trips have realised PnL."""

from collections import deque
from typing import Iterable, List, Tuple

Fill = Tuple[float, float]  # (signed quantity: buy > 0, sell < 0; price)


def round_trip_pnls(fills: Iterable[Fill]) -> List[float]:
    """FIFO-matched realised PnL, one entry per closing fill (long and short)."""
    open_lots: deque = deque()  # [signed qty, price]
    pnls = []
    for qty, price in fills:
        realised = 0.0
        closed_any = False
        while qty and open_lots and (open_lots[0][0] > 0) != (qty > 0):
            lot = open_lots[0]
            matched = min(abs(qty), abs(lot[0]))
            direction = 1.0 if lot[0] > 0 else -1.0
            realised += direction * matched * (price - lot[1])
            lot[0] -= direction * matched
            qty += direction * matched
            closed_any = True
            if abs(lot[0]) < 1e-12:
                open_lots.popleft()
        if abs(qty) > 1e-12:
            open_lots.append([qty, price])
        if closed_any:
            pnls.append(realised)
    return pnls


def compute_win_rate(pnls: List[float]) -> float:
    if not pnls:
        return float("nan")
    return sum(p > 0 for p in pnls) / len(pnls)
