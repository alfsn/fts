# src/trading_bot/runs/records.py
"""What callers see of a run: plain Pydantic records, never ORM rows."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from ..core.enums import OrderSide

RunKind = Literal["backtest", "live"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunHeader(_Record):
    """The run card."""

    run_id: str
    kind: RunKind
    strategy_name: Optional[str] = None
    instrument_id: Optional[str] = None
    initial_cash: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class RunMarket(_Record):
    """One instrument and strategy a run made predictions for. A run can
    have several; its card only names the first."""

    run_id: str
    instrument_id: str
    strategy_name: str


class Segment(_Record):
    """Which model traded the run from ``start_time`` to ``end_time``."""

    model_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class EquityPoint(_Record):
    timestamp: datetime
    cash: float
    position: float
    close: float
    equity: float


class Fill(_Record):
    """One fill at the bar time it happened."""

    order_id: Optional[str] = None
    instrument_id: str
    side: OrderSide
    outcome: Optional[str] = None
    quantity: float
    fill_price: float
    fill_timestamp: datetime


class Prediction(_Record):
    timestamp: datetime
    instrument_id: str
    strategy_name: str
    prediction_output: Optional[str] = None
    predicted_signal: str
    confidence: float
    actual_future_return: Optional[float] = None


class RunRecord(_Record):
    """Everything saved for one run."""

    header: RunHeader
    segments: List[Segment] = []
    equity: List[EquityPoint] = []
    fills: List[Fill] = []
    predictions: List[Prediction] = []

    @property
    def model_id(self) -> Optional[str]:
        """The model of a single-segment run; None when there are several."""
        ids = {s.model_id for s in self.segments}
        return ids.pop() if len(ids) == 1 else None
