# src/trading_bot/execution/handlers/simulated_handler.py

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional

from ...core.enums import OrderSide, OrderStatus
from ...core.schemas import ExecutionResult, IngestionEngineOutput, OrderRequest
from ..abc import BaseExecutionHandler

logger = logging.getLogger(__name__)


# --- Decoupled Delay & Slippage Interfaces ---


class ExecuteDelay(ABC):
    """
    Abstract Base Class for determining trade execution delays.
    """

    @abstractmethod
    def calculate_execution_tick(self, current_tick_index: int) -> int:
        """
        Determines the target tick/bar index at which an order should execute.

        :param current_tick_index: The tick index when the signal is generated.
        :return: The target tick index when the order should be executed.
        """
        pass


class KBarExecuteDelay(ExecuteDelay):
    """
    Delays execution by a fixed number of bars/ticks (T+k shift).
    """

    def __init__(self, k: int = 1) -> None:
        """
        Initializes the KBar delay model.

        :param k: The number of bars/ticks to delay. Must be >= 1.
        """
        if k < 1:
            raise ValueError(f"Execution delay shift k must be >= 1, got {k}")
        self.k = k

    def calculate_execution_tick(self, current_tick_index: int) -> int:
        return current_tick_index + self.k


class PriceSlip(ABC):
    """
    Abstract Base Class for applying slippage and execution price penalties.
    """

    @abstractmethod
    def apply_slippage(self, order: OrderRequest, base_price: float) -> float:
        """
        Calculates the execution price after applying slippage/fees.

        :param order: The original OrderRequest.
        :param base_price: The baseline price of the current execution bar.
        :return: The adjusted average fill price.
        """
        pass


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


# --- Simulated Execution Handler ---


class SimulatedExecutionHandler(BaseExecutionHandler):
    """
    A simulated execution handler that delays fills by T+k ticks
    and applies a price penalty (slippage/fees).
    """

    def __init__(
        self,
        delay_model: ExecuteDelay,
        slippage_model: PriceSlip,
        execution_price_source: str = "close",
        initial_balances: Optional[Mapping[str, float]] = None,
    ) -> None:
        """
        Initializes the SimulatedExecutionHandler.

        :param delay_model: The delay strategy object.
        :param slippage_model: The slippage/fee penalty strategy object.
        :param execution_price_source: Price source to execute on ("open" or "close").
        :param initial_balances: Initial balances dictionary.
        """
        self.delay_model = delay_model
        self.slippage_model = slippage_model

        if execution_price_source not in ("open", "close"):
            raise ValueError(
                f"execution_price_source must be 'open' or 'close', got '{execution_price_source}'"
            )
        self.execution_price_source = execution_price_source

        self._balances = (
            dict(initial_balances) if initial_balances else {"USD": 10000.0}
        )

        self._market_tick_counts: Dict[str, int] = {}
        self._latest_bars: Dict[str, any] = {}
        self._pending_orders: Dict[str, dict] = {}
        self._order_results: Dict[str, ExecutionResult] = {}

    @property
    def market_name(self) -> str:
        return "simulated"

    @property
    def is_simulated(self) -> bool:
        return True

    def execute_order(self, order: OrderRequest) -> ExecutionResult:
        """
        Queues the order for future execution based on the delay model.
        """
        order_id = f"sim-{uuid.uuid4().hex[:8]}"
        market_id = order.market_id

        current_tick = self._market_tick_counts.get(market_id, 0)
        target_tick = self.delay_model.calculate_execution_tick(current_tick)

        self._pending_orders[order_id] = {
            "order": order,
            "execution_tick_index": target_tick,
        }

        logger.info(
            f"[SimulatedExecutionHandler] Queued order {order_id} ({order.side.value}) "
            f"at tick {current_tick}, scheduled to fill at tick {target_tick}."
        )

        return ExecutionResult(
            order_id=order_id,
            status=OrderStatus.OPEN,
            filled_size=0.0,
            avg_price=0.0,
            timestamp=datetime.now(timezone.utc),
            order_type=order.order_type,
        )

    def cancel_order(self, order_id: str) -> ExecutionResult:
        """
        Attempts to cancel a pending delayed order.
        """
        if order_id in self._pending_orders:
            self._pending_orders.pop(order_id)
            result = ExecutionResult(
                order_id=order_id,
                status=OrderStatus.CANCELLED,
                filled_size=0.0,
                avg_price=0.0,
                timestamp=datetime.now(timezone.utc),
            )
            self._order_results[order_id] = result
            logger.info(
                f"[SimulatedExecutionHandler] Cancelled pending order {order_id} successfully."
            )
            return result
        elif order_id in self._order_results:
            return self._order_results[order_id]
        else:
            raise KeyError(f"Order {order_id} not found.")

    def get_order_status(self, order_id: str) -> ExecutionResult:
        """
        Returns the execution status of the given order.
        """
        if order_id in self._order_results:
            return self._order_results[order_id]
        elif order_id in self._pending_orders:
            return ExecutionResult(
                order_id=order_id,
                status=OrderStatus.OPEN,
                filled_size=0.0,
                avg_price=0.0,
                timestamp=datetime.now(timezone.utc),
            )
        else:
            raise KeyError(f"Order {order_id} not found.")

    def get_account_balances(self) -> Mapping[str, float]:
        return self._balances

    def on_tick(
        self, ingestion_output: IngestionEngineOutput, db: Optional[any] = None
    ) -> None:
        """
        Advances the state of markets and executes any due orders.
        """
        # 1. Update tick counters and latest bars
        for market_id, market_data in ingestion_output.market_data.items():
            if market_data.recent_bars:
                latest_bar = market_data.recent_bars[-1]
                self._latest_bars[market_id] = latest_bar
                self._market_tick_counts[market_id] = (
                    self._market_tick_counts.get(market_id, 0) + 1
                )

        # 2. Check pending orders and fill those that have reached their target tick
        for order_id in list(self._pending_orders.keys()):
            pending = self._pending_orders[order_id]
            order = pending["order"]
            target_tick = pending["execution_tick_index"]
            market_id = order.market_id

            current_tick = self._market_tick_counts.get(market_id, 0)
            if current_tick >= target_tick:
                latest_bar = self._latest_bars.get(market_id)
                if not latest_bar:
                    logger.warning(
                        f"Cannot fill order {order_id} - no bar data available for market {market_id}"
                    )
                    continue

                # Determine base execution price
                if self.execution_price_source == "open":
                    base_price = latest_bar.open
                else:
                    base_price = latest_bar.close

                # Apply penalty/slippage
                executed_price = self.slippage_model.apply_slippage(order, base_price)

                # Store filled execution result
                result = ExecutionResult(
                    order_id=order_id,
                    status=OrderStatus.FILLED,
                    filled_size=order.size,
                    avg_price=executed_price,
                    timestamp=latest_bar.timestamp,
                    order_type=order.order_type,
                )
                self._order_results[order_id] = result
                self._pending_orders.pop(order_id)

                # Keep local simulated balances updated
                cost = order.size * executed_price
                if order.side == OrderSide.BUY:
                    self._balances["USD"] = self._balances.get("USD", 0.0) - cost
                    self._balances[market_id] = (
                        self._balances.get(market_id, 0.0) + order.size
                    )
                else:
                    self._balances["USD"] = self._balances.get("USD", 0.0) + cost
                    self._balances[market_id] = (
                        self._balances.get(market_id, 0.0) - order.size
                    )

                logger.info(
                    f"[SimulatedExecutionHandler] Filled order {order_id} at price {executed_price:.4f} "
                    f"(base price: {base_price:.4f}, side: {order.side.value})."
                )
