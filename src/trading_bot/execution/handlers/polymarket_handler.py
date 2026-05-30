# src/trading_bot/execution/handlers/polymarket_handler.py

import logging
import uuid
from datetime import datetime, timezone
from typing import Mapping

from ...core.enums import OrderStatus
from ...core.schemas import ExecutionResult, OrderRequest
from ..abc import BaseExecutionHandler

logger = logging.getLogger(__name__)


class PolymarketHandler(BaseExecutionHandler):
    """
    A concrete mock implementation of BaseExecutionHandler.
    Simulates placing, canceling, and checking order statuses on Polymarket.
    """

    @property
    def market_name(self) -> str:
        return "polymarket"

    def execute_order(self, order: OrderRequest) -> ExecutionResult:
        """Simulates placing an order and instantly filling it for simple execution flow."""
        order_id = f"poly-{uuid.uuid4().hex[:8]}"
        logger.info(
            f"[PolymarketHandler] Mock order submitted. "
            f"ID: {order_id}, side: {order.side.value}, size: {order.size:.4f} shares."
        )
        return ExecutionResult(
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_size=order.size,
            avg_price=order.price,
            timestamp=datetime.now(timezone.utc),
            order_type=order.order_type,
        )

    def cancel_order(self, order_id: str) -> ExecutionResult:
        """Simulates canceling an open order."""
        logger.info(
            f"[PolymarketHandler] Mock cancel request sent for order: {order_id}"
        )
        return ExecutionResult(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            filled_size=0.0,
            avg_price=0.0,
            timestamp=datetime.now(timezone.utc),
        )

    def get_order_status(self, order_id: str) -> ExecutionResult:
        """Simulates retrieving the status of an order."""
        logger.info(
            f"[PolymarketHandler] Mock get status request sent for order: {order_id}"
        )
        return ExecutionResult(
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_size=1.0,
            avg_price=1.0,
            timestamp=datetime.now(timezone.utc),
        )

    def get_account_balances(self) -> Mapping[str, float]:
        """Returns mock wallet balances."""
        return {
            "USD": 10000.0,
            "USDC": 5000.0,
            "MATIC": 150.0,
        }
