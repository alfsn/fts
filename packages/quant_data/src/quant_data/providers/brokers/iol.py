from datetime import datetime, timezone

from quant_core.enums import Currency
from quant_core.models import CashBalance, Portfolio, Position

from .base import BrokerConnector


class IOLConnector(BrokerConnector):
    @property
    def name(self) -> str:
        return "IOL"

    def get_portfolio(self) -> Portfolio:
        # Mock implementation
        return Portfolio(
            timestamp=datetime.now(timezone.utc),
            positions=[
                Position(
                    instrument_id="YPFD",
                    currency=Currency.ARS,
                    quantity=200.0,
                    cost_basis=15000.0,
                    current_price=16000.0,
                )
            ],
            cash_balances=[
                CashBalance(currency=Currency.ARS, total=200000.0, available=200000.0)
            ],
        )
