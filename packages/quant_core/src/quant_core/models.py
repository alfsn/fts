from datetime import datetime
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from .enums import (
    AssetType,
    BarType,
    Geography,
    OrderSide,
    OrderStatus,
    OrderType,
    TransactionType,
)

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
    outcome: Optional[str] = None
    quantity: float
    cost_basis: float = Field(..., ge=0)
    current_price: Optional[float] = Field(None, ge=0)


class CashBalance(BaseModel):
    currency: str
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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    positions: List[Position]
    cash_balances: List[CashBalance]
    open_orders: List[OrderRequest] = Field(default_factory=list)


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
