from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from .enums import (
    AssetType,
    BarType,
    CorporateActionType,
    Currency,
    Geography,
    OrderSide,
    OrderStatus,
    OrderType,
    RateType,
    TransactionType,
)


class MissingFXRateError(Exception):
    pass


class FXContext(BaseModel):
    base_currency: Currency
    rates: Mapping[Currency, float] = Field(default_factory=dict)

    def get_rate(self, target_currency: Currency) -> float:
        if target_currency == self.base_currency:
            return 1.0

        if target_currency not in self.rates:
            raise MissingFXRateError(
                f"Missing FX rate to convert {target_currency.value} to {self.base_currency.value}"
            )

        return self.rates[target_currency]


# --- 1. Instrument Taxonomy (Composition) ---


class TradFiDetails(BaseModel):
    type: Literal["tradfi"] = "tradfi"
    asset_type: AssetType
    geography: Geography
    industry: str
    currency: str


class PredictionMarketDetails(BaseModel):
    type: Literal["prediction"] = "prediction"
    end_date: datetime
    resolution_source: str


InstrumentDetails = Annotated[
    Union[TradFiDetails, PredictionMarketDetails], Field(discriminator="type")
]


class Instrument(BaseModel):
    instrument_id: str
    name: str
    details: InstrumentDetails


# --- 2. Live & Historical Holdings ---


class Position(BaseModel):
    instrument_id: str
    currency: Currency = Field(default=Currency.USD)
    outcome: Optional[str] = None
    quantity: float
    cost_basis: float = Field(..., ge=0)
    current_price: Optional[float] = Field(None, ge=0)
    # Market-data symbol when it differs from the broker's id (e.g. YPF in ARS -> YPFD.BA).
    provider_symbol: Optional[str] = None

    @property
    def market_symbol(self) -> str:
        return self.provider_symbol or self.instrument_id

    def market_value(self, fx_context: "FXContext") -> float:
        """In the base currency. An unpriced position is NaN, never 0."""
        price = float("nan") if self.current_price is None else self.current_price
        return price * self.quantity * fx_context.get_rate(self.currency)


class CashBalance(BaseModel):
    currency: Currency
    total: float
    available: float


class OrderRequest(BaseModel):
    instrument_id: str
    side: OrderSide
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    order_type: OrderType = Field(OrderType.LIMIT)
    outcome: Optional[str] = None


# A listing is one instrument in one currency: YPF in ARS and YPF in USD are
# different holdings with different prices (PR 2b, D13).
Listing = Tuple[str, Currency]


def listing_symbols(items: Iterable["Position | Transaction"]) -> Dict[Listing, str]:
    """One market symbol per listing across positions and ledger lines."""
    symbols: Dict[Listing, str] = {}
    for item in items:
        if item.instrument_id is None:
            continue
        key = (item.instrument_id, item.currency)
        symbol = item.market_symbol
        if symbols.setdefault(key, symbol) != symbol:
            raise ValueError(
                f"Listing {key[0]} in {key[1].value} maps to two market symbols: "
                f"{symbols[key]} and {symbol}."
            )
    return symbols


class Portfolio(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    base_currency: Currency = Field(default=Currency.USD)
    positions: List[Position]
    cash_balances: List[CashBalance]
    open_orders: List[OrderRequest] = Field(default_factory=list)
    # Hand-written ledger (PR 2b). Empty means "snapshot only".
    transactions: List["Transaction"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ledger_is_consistent(self) -> "Portfolio":
        listing_symbols([*self.positions, *self.transactions])
        if self.transactions:
            first_day = min(t.date for t in self.transactions)
            opening = {TransactionType.DEPOSIT, TransactionType.TRANSFER_IN}
            if not any(
                t.type in opening for t in self.transactions if t.date == first_day
            ):
                raise ValueError(
                    "The ledger must start with a deposit or transfer_in: the first "
                    "valued day needs something to value."
                )
        return self

    def total_value(self, fx_context: "FXContext") -> float:
        """Market value in the base currency; NaN if any position is unpriced."""
        val = 0.0
        for p in self.positions:
            val += p.market_value(fx_context)
        for cb in self.cash_balances:
            rate = fx_context.get_rate(cb.currency)
            val += cb.total * rate
        return val


# --- 3. Ledger & Operational Logs ---


_TRADES = {TransactionType.BUY, TransactionType.SELL}
_TRANSFERS = {TransactionType.TRANSFER_IN, TransactionType.TRANSFER_OUT}
_CASH_ONLY = {TransactionType.DEPOSIT, TransactionType.WITHDRAWAL}


class Transaction(BaseModel):
    """
    One hand-typed ledger line (PR 2b). Every number is positive; ``type``
    gives the direction. ``date`` is the exchange's local trade date (the pay
    date for a dividend); the event is in that day's close.
    """

    model_config = ConfigDict(extra="forbid")

    date: date
    type: TransactionType
    currency: Currency
    instrument_id: Optional[str] = None
    # Market-data symbol when it differs from the broker's id (as Position).
    provider_symbol: Optional[str] = None
    quantity: Optional[float] = Field(None, gt=0)  # shares
    price: Optional[float] = Field(None, gt=0)  # raw, per share, in `currency`
    amount: Optional[float] = Field(None, gt=0)  # cash, in `currency`
    fees: float = Field(0.0, ge=0)  # commission on a trade, in `currency`
    note: Optional[str] = None

    @property
    def market_symbol(self) -> str:
        return self.provider_symbol or self.instrument_id

    @property
    def listing(self) -> "Listing":
        return (self.instrument_id, self.currency)

    @model_validator(mode="after")
    def _fields_per_type(self) -> "Transaction":
        t = self.type
        if t is TransactionType.SPLIT:
            raise ValueError(
                "split lines are not accepted: splits come from the cached corporate "
                "actions (PR 2a), so typing them would count them twice."
            )
        if t is TransactionType.COUPON:
            raise ValueError(
                "coupon lines are not accepted yet: bonds have no price source (PR 2c)."
            )
        if t in _TRADES:
            required, forbidden = ("instrument_id", "quantity", "price"), ("amount",)
        elif t in _TRANSFERS:
            required, forbidden = ("instrument_id", "quantity"), ("price", "amount")
        elif t in _CASH_ONLY:
            required = ("amount",)
            forbidden = ("instrument_id", "quantity", "price")
        elif t is TransactionType.DIVIDEND:
            required, forbidden = ("instrument_id", "amount"), ("quantity", "price")
        else:  # FEE: may name the instrument it relates to
            required, forbidden = ("amount",), ("quantity", "price")
        for field in required:
            if getattr(self, field) is None:
                raise ValueError(f"{t.value} needs {field}.")
        for field in forbidden:
            if getattr(self, field) is not None:
                raise ValueError(f"{t.value} must not have {field}.")
        if t not in _TRADES and self.fees:
            raise ValueError(f"{t.value} must not have fees; type a fee line instead.")
        if self.provider_symbol and self.instrument_id is None:
            raise ValueError("provider_symbol needs instrument_id.")
        return self


class ExecutionResult(BaseModel):
    order_id: str
    status: OrderStatus
    filled_quantity: float = Field(..., ge=0)
    avg_price: float = Field(..., ge=0)
    timestamp: datetime
    order_type: OrderType = Field(OrderType.LIMIT)


# --- 4. Market Data (Data Ingestion) ---


class PriceLevel(BaseModel):
    price: float = Field(..., gt=0)
    quantity: float = Field(..., ge=0)


class OrderBook(BaseModel):
    bids: List[PriceLevel]
    asks: List[PriceLevel]


class Trade(BaseModel):
    price: float = Field(..., gt=0)
    quantity: float = Field(..., gt=0)
    timestamp: datetime
    side: OrderSide
    outcome: Optional[str] = None


class BarData(BaseModel):
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    bar_type: BarType
    interval: Optional[str] = None
    ticks_count: int = Field(..., ge=1)
    dollar_volume: float = Field(..., ge=0)


class MarketData(BaseModel):
    instrument_id: str
    order_book: Optional[OrderBook] = None
    recent_trades: Optional[List[Trade]] = None
    details: Instrument
    recent_bars: List[BarData] = Field(default_factory=list)


class MarketDataRequest(BaseModel):
    instrument_id: str
    interval: str
    count: int = 100
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_order_book: bool = True
    include_trades: bool = True

    def resolve_bounds(self) -> tuple[datetime, datetime]:
        import re
        from datetime import timedelta

        end = self.end_date or datetime.now(timezone.utc)
        if self.start_date:
            return self.start_date, end

        match = re.match(r"(\d+)([a-zA-Z]+)", self.interval)
        if not match:
            return end - timedelta(days=self.count), end

        val, unit = int(match.group(1)), match.group(2).lower()
        if unit in ["m", "min"]:
            delta = timedelta(minutes=val * self.count * 1.1)
        elif unit in ["h", "hr"]:
            delta = timedelta(hours=val * self.count * 1.5)
        elif unit in ["d", "day"]:
            delta = timedelta(days=int(val * self.count * 1.5))
        elif unit in ["w", "wk"]:
            delta = timedelta(weeks=val * self.count)
        else:
            delta = timedelta(days=self.count)

        return end - delta, end


class RateLimit(BaseModel):
    """
    A provider's declared request budget, enforced in one place
    (``quant_data.providers.throttle``): at most one call per
    ``min_interval_s``, and on a rate-limit error ``max_retries`` retries with
    exponential backoff ``backoff_s * 2**k``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_interval_s: float = Field(..., ge=0)
    max_retries: int = Field(..., ge=0)
    backoff_s: float = Field(..., ge=0)


# --- 5. Additional Domain Models ---


class FXRate(BaseModel):
    base_currency: Currency
    quote_currency: Currency
    rate_type: RateType
    rate: float = Field(..., gt=0)
    timestamp: datetime


class CorporateAction(BaseModel):
    """
    A split or dividend as the vendor reports it, in raw (as-paid) terms.

    ``ex_date``: the first session on which the shares trade without the
    entitlement. Only its calendar date in the exchange's local zone matters;
    bars on or after it are in new share units. ``ratio``: new shares per old
    share (10:1 split -> 10.0; 1:10 reverse split -> 0.1), as ``yf.Ticker.splits``
    reports it. ``amount``: cash per *old* share, before any later split.
    """

    model_config = ConfigDict(extra="forbid")

    instrument_id: str
    action_type: CorporateActionType
    ex_date: datetime
    record_date: Optional[datetime] = None
    payable_date: Optional[datetime] = None
    amount: Optional[float] = Field(None, ge=0)
    ratio: Optional[float] = Field(None, gt=0)

    @model_validator(mode="after")
    def _required_fields_per_type(self) -> "CorporateAction":
        if self.action_type is CorporateActionType.SPLIT and self.ratio is None:
            raise ValueError("A split needs a ratio (new shares per old share).")
        if self.action_type is CorporateActionType.DIVIDEND and self.amount is None:
            raise ValueError("A dividend needs an amount per share.")
        return self


class DrawdownEpisode(BaseModel):
    """A peak-to-recovery excursion below the high-water mark."""

    peak: datetime
    trough: datetime
    recovery: Optional[datetime] = Field(
        None, description="First new high after the trough; None if still underwater."
    )
    depth: float = Field(..., le=0, description="Trough drawdown as a fraction (<= 0).")
    duration_days: float = Field(
        ..., ge=0, description="Peak to recovery (or to last observation)."
    )


class PerformanceMetrics(BaseModel):
    """
    Backward-looking KPIs. Returns are decimal fractions (0.05 == 5%);
    percentage formatting is a presentation concern. Statistics that are
    undefined for the input (e.g. Sharpe on < 2 daily returns) are NaN, never 0.0.
    """

    model_config = ConfigDict(extra="forbid")

    total_return: float = Field(float("nan"), description="Cumulative TWR.")
    cagr: float = Field(float("nan"), description="Annualised TWR.")
    volatility: float = Field(float("nan"), description="Annualised volatility.")
    sharpe_ratio: float = Field(float("nan"), description="Annualised Sharpe ratio.")
    sortino_ratio: float = Field(float("nan"), description="Annualised Sortino ratio.")
    calmar_ratio: float = Field(float("nan"), description="CAGR / |max drawdown|.")
    max_drawdown: float = Field(
        float("nan"), description="Maximum peak-to-trough drawdown (<= 0)."
    )
    time_underwater_days: float = Field(
        float("nan"), description="Longest time below the high-water mark (days)."
    )
    win_rate: float = Field(
        float("nan"), description="Fraction of closed round trips with positive PnL."
    )
    total_trades: int = Field(0, description="Total filled trade executions count.")
