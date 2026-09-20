# src/trading_bot/execution/slippage.py

from ..core.enums import OrderSide
from ..core.schemas import OrderRequest
from .abc import PriceSlip


class FlatPriceSlip(PriceSlip):
    """
    Applies a fixed basis point percentage penalty to the execution price.
    """

    def __init__(self, slippage_pct: float = 0.0005) -> None:
        """
        Initializes the FlatPriceSlip model.

        :param slippage_pct: The fixed penalty percentage (e.g. 0.0005 for 0.05%).
        """
        self.slippage_pct = slippage_pct

    def apply_slippage(self, order: OrderRequest, base_price: float) -> float:
        if order.side == OrderSide.BUY:
            return base_price * (1.0 + self.slippage_pct)
        else:
            return base_price * (1.0 - self.slippage_pct)
