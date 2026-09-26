from datetime import datetime, timezone
from typing import List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, Field
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


class Portfolio(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    base_currency: Currency = Field(default=Currency.USD)
    positions: List[Position]
    cash_balances: List[CashBalance]
    open_orders: List[OrderRequest] = Field(default_factory=list)

    def total_value(self, fx_context: "FXContext") -> float:
        val = 0.0
        for p in self.positions:
            rate = fx_context.get_rate(p.currency)
            val += (p.current_price or 0.0) * p.quantity * rate
        for cb in self.cash_balances:
            rate = fx_context.get_rate(cb.currency)
            val += cb.total * rate
        return val


# --- 3. Ledger & Operational Logs ---


class Transaction(BaseModel):
    transaction_id: str
    instrument_id: str
    related_instrument_id: Optional[str] = None
    transaction_type: TransactionType
    quantity: float
    price: float = Field(..., ge=0)
    timestamp: datetime


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


# --- 5. Additional Domain Models ---


class FXRate(BaseModel):
    base_currency: Currency
    quote_currency: Currency
    rate_type: RateType
    rate: float = Field(..., gt=0)
    timestamp: datetime


class CorporateAction(BaseModel):
    instrument_id: str
    action_type: CorporateActionType
    ex_date: datetime
    record_date: Optional[datetime] = None
    payable_date: Optional[datetime] = None
    amount: Optional[float] = None
    ratio: Optional[float] = None
