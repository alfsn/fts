from enum import Enum


class TransactionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    DIVIDEND = "dividend"
    SPLIT = "split"
    COUPON = "coupon"


class AssetType(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    CASH = "cash"
    PREDICTION = "prediction"


class Geography(str, Enum):
    US = "us"
    ARGENTINA = "argentina"
    DEVELOPED_EX_US = "developed_ex_us"
    EMERGING_MARKETS = "emerging_markets"
    GLOBAL = "global"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    FLAT = "flat"


class BarType(str, Enum):
    TIME = "time"
    VOLUME = "volume"
    DOLLAR = "dollar"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SizingStrategyType(str, Enum):
    KELLY_CRITERION = "kelly_criterion"
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"


class Currency(str, Enum):
    USD = "USD"
    ARS = "ARS"
    EUR = "EUR"


class RateType(str, Enum):
    MEP = "MEP"
    CCL = "CCL"
    OFFICIAL = "OFFICIAL"


class CorporateActionType(str, Enum):
    SPLIT = "split"
    DIVIDEND = "dividend"
    MERGER = "merger"
    SPINOFF = "spinoff"
