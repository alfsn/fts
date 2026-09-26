from datetime import datetime, timezone

from quant_core.enums import Currency
from quant_core.models import CashBalance, Portfolio, Position

from .base import BrokerConnector


class InviuConnector(BrokerConnector):
    @property
    def name(self) -> str:
        return "Inviu"

    def get_portfolio(self) -> Portfolio:
        # Mock implementation
        return Portfolio(
            timestamp=datetime.now(timezone.utc),
            positions=[
                Position(
                    instrument_id="GGAL",
                    currency=Currency.ARS,
                    quantity=500.0,
                    cost_basis=2000.0,
                    current_price=2100.0,
                )
            ],
            cash_balances=[
                CashBalance(currency=Currency.ARS, total=500000.0, available=500000.0)
            ],
        )
