from datetime import datetime, timezone

from quant_core.enums import Currency
from quant_core.models import CashBalance, Portfolio, Position

from .base import BrokerConnector


class IBKRConnector(BrokerConnector):
    @property
    def name(self) -> str:
        return "IBKR"

    def get_portfolio(self) -> Portfolio:
        # Mock implementation
        return Portfolio(
            timestamp=datetime.now(timezone.utc),
            positions=[
                Position(
                    instrument_id="AAPL",
                    currency=Currency.USD,
                    quantity=100.0,
                    cost_basis=150.0,
                    current_price=175.0,
                )
            ],
            cash_balances=[
                CashBalance(currency=Currency.USD, total=10000.0, available=10000.0)
            ],
        )
