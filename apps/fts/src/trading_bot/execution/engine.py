# src/trading_bot/execution/engine.py

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..core.schemas import (
    ExecutionResult,
    IngestionEngineOutput,
    OrderRequest,
    OrderStatus,
)
from ..risk_management.portfolio import Portfolio
from ..runs import RunRecorder
from .abc import BaseExecutionHandler

logger = logging.getLogger(__name__)


class ExecutionErrorType(Enum):
    """Types of execution errors for better error handling"""

    HANDLER_FAILURE = "handler_failure"
    DATABASE_FAILURE = "database_failure"
    PORTFOLIO_FAILURE = "portfolio_failure"
    TIMEOUT = "timeout"
    INVALID_ORDER = "invalid_order"


# Statuses after which the exchange never changes an order again
_FINAL = (
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.FAILED,
)


class ExecutionEngine:
    """
    Orchestrates order execution, logging, and portfolio updates.

    Enhanced with robust error handling, state reconciliation,
    and recovery mechanisms.
    """

    def __init__(
        self,
        execution_handler: BaseExecutionHandler,
        portfolio: Portfolio,
        recorder: RunRecorder,
        max_retry_attempts: int = 3,
        enable_auto_reconciliation: bool = True,
    ) -> None:
        """
        Initializes the engine with error handling configuration.

        :param execution_handler: BaseExecutionHandler implementation
        :param portfolio: The bot's shared Portfolio object
        :param recorder: Saves this run's orders and fills (``trading_bot.runs``)
        :param max_retry_attempts: Number of retries for transient failures
        :param enable_auto_reconciliation: Enable automatic state reconciliation
        """
        self.handler = execution_handler
        self.portfolio = portfolio
        self.max_retry_attempts = max_retry_attempts
        self.enable_auto_reconciliation = enable_auto_reconciliation
        self.recorder = recorder

        # Track pending reconciliation items
        self._reconciliation_queue: List[Any] = []

        logger.info(
            f"ExecutionEngine initialized with handler for: "
            f"{self.handler.market_name}, "
            f"max_retries={max_retry_attempts}, "
            f"auto_reconciliation={enable_auto_reconciliation}"
        )

    def execute_order(
        self,
        order: OrderRequest,
        db: Session,
        strategy_name: str,
    ) -> ExecutionResult:
        """
        Main entry point to execute a new trade order with robust error handling.

        :param order: The OrderRequest object from the RiskManager
        :param db: The SQLAlchemy database session
        :param strategy_name: The name of the strategy that generated the signal
        :return: An ExecutionResult object with the final status
        """
        logger.info(
            f"[{strategy_name}] Executing order for {order.instrument_id}: "
            f"{order.side.value} {order.quantity:.4f} shares @ ${order.price:.4f}"
        )

        result: Optional[ExecutionResult] = None

        # Phase 1: Execute order via handler
        try:
            result = self.handler.execute_order(order)
            logger.info(
                f"Handler returned result for {order.instrument_id}: "
                f"ID: {result.order_id}, Status: {result.status}"
            )

        except Exception as e:
            logger.error(
                f"Execution API call failed for {order.instrument_id}: {e}",
                exc_info=True,
            )
            # Create a failed result but DON'T proceed with portfolio updates
            # since we don't know if the order was actually placed
            result = self._create_failed_result(
                error_message=str(e), error_type=ExecutionErrorType.HANDLER_FAILURE
            )

        # Phase 2: Database and portfolio updates (critical section)
        if result:
            try:
                # Log order to database and commit
                self.recorder.record_order(db, order, result, strategy_name)
                db.commit()

                # Only update portfolio if DB logging succeeded
                self._safe_portfolio_update(db, order, result)
                db.commit()

            except SQLAlchemyError as e:
                logger.critical(
                    f"DATABASE ERROR: Failed to log order {result.order_id}. "
                    f"Order may be live on exchange but not tracked! Error: {e}",
                    exc_info=True,
                )
                db.rollback()

                # Queue for reconciliation if enabled
                if self.enable_auto_reconciliation:
                    self._queue_for_reconciliation(
                        result.order_id, order, "db_write_failed", strategy_name
                    )

            except Exception as e:
                logger.critical(
                    f"PORTFOLIO UPDATE ERROR: Failed to update portfolio for "
                    f"order {result.order_id}. Database inconsistent! Error: {e}",
                    exc_info=True,
                )
                # Portfolio is out of sync - requires manual intervention
                self._handle_portfolio_desync(result.order_id, order, result)

        return result

    def check_order_status(self, order_id: str, db: Session) -> ExecutionResult:
        """
        Polls the handler for the status of a specific open order.
        Enhanced with reconciliation logic.

        :param order_id: The unique ID of the order to check
        :param db: The SQLAlchemy database session
        :return: An ExecutionResult with the latest status
        """
        logger.debug(f"Checking status for order {order_id}")

        try:
            # Get status from exchange
            result = self.handler.get_order_status(order_id)

            # Verify database state
            logged_status = self.recorder.order_status(db, order_id)

            if logged_status is None:
                logger.warning(
                    f"Order {order_id} exists on exchange but not in database! "
                    f"This indicates a previous logging failure."
                )
                # Attempt to create missing log entry if we have enough info
                if self.enable_auto_reconciliation:
                    self._reconcile_missing_order(db, order_id, result)
                return result

            # Update if the status changed, or if the order is final but still
            # open in the Portfolio (a fill whose commit failed earlier)
            unapplied = result.status in _FINAL and order_id in getattr(
                self.portfolio, "open_orders", {}
            )
            if logged_status != result.status or unapplied:
                logger.info(
                    f"Status for order {order_id} changed: "
                    f"{logged_status.value} -> {result.status.value}"
                )

                try:
                    # Log change to database
                    self.recorder.update_order(db, result)

                    # Update portfolio after successful DB update
                    self._safe_portfolio_update(db, None, result)

                    # Commit both atomically
                    db.commit()

                except SQLAlchemyError as e:
                    # The order stays open in the Portfolio: retried next check
                    logger.error(
                        f"Failed to update order log for {order_id}: {e}", exc_info=True
                    )
                    db.rollback()

            return result

        except Exception as e:
            logger.error(f"Failed to get status for {order_id}: {e}", exc_info=True)
            # Return a synthetic result to signal the error
            return self._create_synthetic_result_for_id(order_id, OrderStatus.FAILED)

    def cancel_order(self, order_id: str, db: Session) -> ExecutionResult:
        """
        Attempts to cancel an open order with proper error handling.

        :param order_id: The unique ID of the order to cancel
        :param db: The SQLAlchemy database session
        :return: An ExecutionResult
        """
        logger.info(f"Attempting to cancel order {order_id}")

        try:
            # Attempt cancellation via handler
            result = self.handler.cancel_order(order_id)

        except Exception as e:
            # The order may still be live: leave its row and the Portfolio as
            # they are, so the next status check still tracks it.
            logger.error(f"Cancel API call for {order_id} failed: {e}", exc_info=True)
            return self._create_synthetic_result_for_id(order_id, OrderStatus.FAILED)

        # Update database and portfolio
        if result:
            try:
                self.recorder.update_order(db, result)
                self._safe_portfolio_update(db, None, result)
                db.commit()

            except Exception as e:
                logger.critical(
                    f"CRITICAL: Failed to update state after cancel attempt "
                    f"for {order_id}. Error: {e}",
                    exc_info=True,
                )
                db.rollback()

        return result

    # --- Enhanced Helper Methods ---

    def _safe_portfolio_update(
        self, db: Session, order: Optional[OrderRequest], result: ExecutionResult
    ):
        """
        Wrapper for portfolio updates with error isolation.

        :param db: Database session
        :param order: Original order request (can be None for status updates)
        :param result: Execution result
        """
        try:
            self._apply_to_portfolio(db, order, result)
        except SQLAlchemyError:
            # The caller rolls back; nothing was applied to the Portfolio
            raise
        except Exception as e:
            logger.error(
                f"Portfolio update failed for order {result.order_id}: {e}",
                exc_info=True,
            )
            # Don't raise - portfolio can be resynced later
            self._queue_for_reconciliation(result.order_id, order, "portfolio_resync")

    def _apply_to_portfolio(
        self, db: Session, order: Optional[OrderRequest], result: ExecutionResult
    ):
        """Track or fill the order in the Portfolio, saving the fill first.
        Raises on any failure (reconciliation counts it as failed)."""
        if result.status == OrderStatus.OPEN and order:
            self.portfolio.add_open_order(result.order_id, order)
        else:
            if order and isinstance(getattr(self.portfolio, "open_orders", None), dict):
                if result.order_id not in self.portfolio.open_orders:
                    logger.warning(
                        f"Order {result.order_id} was filled (or finalized) immediately "
                        f"without being previously tracked as OPEN. Temporarily tracking it "
                        f"to process the fill."
                    )
                    self.portfolio.add_open_order(result.order_id, order)
            # Fills are the truth: commit the fill (with any pending order
            # row) before the Portfolio applies it. If the commit fails the
            # Portfolio is untouched and the order stays open, so the next
            # status check or reconciliation applies it again.
            to_fill = self.portfolio.fill_for(result)
            if to_fill is not None:
                self.recorder.record_fill(db, to_fill, result)
                db.commit()
            filled = self.portfolio.update_order_status(result)
            if (filled is None) != (to_fill is None):
                raise RuntimeError(
                    f"Order {result.order_id}: the saved fill and the "
                    "Portfolio's applied fill disagree."
                )

    def _create_failed_result(
        self,
        error_message: str,
        error_type: ExecutionErrorType = ExecutionErrorType.HANDLER_FAILURE,
    ) -> ExecutionResult:
        """
        Creates a synthetic ExecutionResult for a failed API call.

        :param error_message: Description of the error
        :param error_type: Type of error that occurred
        :return: ExecutionResult with FAILED status
        """
        # Include error type in the order ID for debugging
        synthetic_id = f"local_fail_{error_type.value}_{uuid.uuid4().hex[:8]}"

        logger.warning(
            f"Creating synthetic failed result: {synthetic_id} - {error_message}"
        )

        return ExecutionResult(
            order_id=synthetic_id,
            status=OrderStatus.FAILED,
            filled_quantity=0,
            avg_price=0,
            timestamp=datetime.now(timezone.utc),
        )

    def _create_synthetic_result_for_id(
        self, order_id: str, status: OrderStatus
    ) -> ExecutionResult:
        """Creates a synthetic result for a known order ID."""
        return ExecutionResult(
            order_id=order_id,
            status=status,
            filled_quantity=0,
            avg_price=0,
            timestamp=datetime.now(timezone.utc),
        )

    # --- Reconciliation Methods ---

    def _queue_for_reconciliation(
        self,
        order_id: str,
        order: Optional[OrderRequest],
        context: str,
        strategy_name: Optional[str] = None,
    ):
        """
        Adds an order to the reconciliation queue (once per order).

        :param order_id: Order ID to reconcile
        :param order: Original order request (if available)
        :param context: Context string for debugging
        :param strategy_name: Strategy that placed the order (if known)
        """
        if any(item["order_id"] == order_id for item in self._reconciliation_queue):
            return
        self._reconciliation_queue.append(
            {
                "order_id": order_id,
                "order": order,
                "context": context,
                "strategy_name": strategy_name,
                "timestamp": datetime.now(timezone.utc),
            }
        )
        logger.warning(
            f"Queued order {order_id} for reconciliation. "
            f"Context: {context}, Queue size: {len(self._reconciliation_queue)}"
        )

    def _reconcile_missing_order(
        self, db: Session, order_id: str, result: ExecutionResult
    ):
        """
        Attempts to create a database entry for an order that exists
        on the exchange but is missing from our database.

        :param db: Database session
        :param order_id: Order ID to reconcile
        :param result: Current execution result from exchange
        """
        logger.warning(f"Attempting to reconcile missing order: {order_id}")

        # This is a best-effort recovery - we're missing the original OrderRequest
        # In production, you might query the exchange API for order details
        try:
            self.recorder.record_unknown_order(db, result)
            db.commit()
            logger.info(f"Successfully reconciled order {order_id}")

        except Exception as e:
            logger.error(f"Failed to reconcile order {order_id}: {e}")
            db.rollback()

    def _handle_portfolio_desync(
        self, order_id: str, order: OrderRequest, result: ExecutionResult
    ):
        """
        Handles the case where portfolio update fails but database succeeded.
        This is a critical state requiring manual intervention.

        :param order_id: The order ID
        :param order: Original order request
        :param result: Execution result
        """
        logger.critical(
            f"PORTFOLIO DESYNC DETECTED for order {order_id}. "
            f"Manual reconciliation required. Order details: "
            f"market={order.instrument_id}, side={order.side.value}, "
            f"quantity={order.quantity}, status={result.status.value}"
        )

        # In production, you would:
        # 1. Send an alert to the operations team
        # 2. Write to a dead letter queue
        # 3. Trigger an automated reconciliation workflow

        self._queue_for_reconciliation(order_id, order, "portfolio_desync")

    def reconcile_all_orders(self, db: Session) -> dict:
        """
        Manually triggers reconciliation of all queued orders.
        Returns a summary of the reconciliation results.

        :param db: Database session
        :return: Dictionary with reconciliation statistics
        """
        logger.info(
            f"Starting reconciliation of {len(self._reconciliation_queue)} orders"
        )

        results: Dict[str, Any] = {
            "total": len(self._reconciliation_queue),
            "success": 0,
            "failed": 0,
            "errors": [],
        }

        for item in self._reconciliation_queue[:]:  # Copy list to allow removal
            try:
                order_id = item["order_id"]

                # Check current status on exchange
                current_result = self.handler.get_order_status(order_id)

                order = item.get("order")
                if order is not None and (
                    self.recorder.order_status(db, order_id) is None
                ):
                    # The first write of this order failed, so the Portfolio
                    # never saw it: record it from the request and apply it.
                    self.recorder.record_order(
                        db,
                        order,
                        current_result,
                        item.get("strategy_name") or "reconciled",
                    )
                    self._apply_to_portfolio(db, order, current_result)
                else:
                    # The order row exists (or the request is unknown): an
                    # unapplied fill is still open in the Portfolio.
                    if not self.recorder.update_order(db, current_result):
                        self.recorder.record_unknown_order(db, current_result)
                    self._apply_to_portfolio(db, None, current_result)
                db.commit()

                results["success"] += 1
                self._reconciliation_queue.remove(item)

            except Exception as e:
                db.rollback()
                results["failed"] += 1
                results["errors"].append(f"{item['order_id']}: {str(e)}")
                logger.error(f"Reconciliation failed for {item['order_id']}: {e}")

        logger.info(
            f"Reconciliation complete. Success: {results['success']}, "
            f"Failed: {results['failed']}"
        )

        return results

    def on_tick(self, ingestion_output: IngestionEngineOutput, db: Session) -> None:
        """
        Called at the start of each pipeline tick to advance the execution
        handler and automatically check the status of any open orders.
        """
        if hasattr(self.handler, "on_tick"):
            self.handler.on_tick(ingestion_output, db)

        # Reconcile open orders in the portfolio
        open_order_ids = list(self.portfolio.open_orders.keys())
        for order_id in open_order_ids:
            self.check_order_status(order_id, db)
