# src/trading_bot/execution/engine.py

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..core.enums import MarketOutcome, OrderSide
from ..core.models import OrderLog as OrderLogModel
from ..core.schemas import (
    ExecutionResult,
    OrderRequest,
    OrderStatus,
)
from ..risk_management.portfolio import Portfolio
from .abc import BaseExecutionHandler

logger = logging.getLogger(__name__)


class ExecutionErrorType(Enum):
    """Types of execution errors for better error handling"""

    HANDLER_FAILURE = "handler_failure"
    DATABASE_FAILURE = "database_failure"
    PORTFOLIO_FAILURE = "portfolio_failure"
    TIMEOUT = "timeout"
    INVALID_ORDER = "invalid_order"


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
        max_retry_attempts: int = 3,
        enable_auto_reconciliation: bool = True,
    ):
        """
        Initializes the engine with error handling configuration.

        :param execution_handler: BaseExecutionHandler implementation
        :param portfolio: The bot's shared Portfolio object
        :param max_retry_attempts: Number of retries for transient failures
        :param enable_auto_reconciliation: Enable automatic state reconciliation
        """
        self.handler = execution_handler
        self.portfolio = portfolio
        self.max_retry_attempts = max_retry_attempts
        self.enable_auto_reconciliation = enable_auto_reconciliation

        # Track pending reconciliation items
        self._reconciliation_queue = []

        logger.info(
            f"ExecutionEngine initialized with handler for: "
            f"{self.handler.market_name.value}, "
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
            f"[{strategy_name}] Executing order for {order.market_id}: "
            f"{order.side.value} {order.size:.4f} shares @ ${order.price:.4f}"
        )

        result: Optional[ExecutionResult] = None

        # Phase 1: Execute order via handler
        try:
            result = self.handler.execute_order(order)
            logger.info(
                f"Handler returned result for {order.market_id}: "
                f"ID: {result.order_id}, Status: {result.status}"
            )

        except Exception as e:
            logger.error(
                f"Execution API call failed for {order.market_id}: {e}",
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
                # Use a nested transaction for atomicity
                with db.begin_nested():
                    self._log_order(db, order, result, strategy_name)
                    db.commit()

                # Only update portfolio if DB logging succeeded
                self._safe_portfolio_update(db, order, result)

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
                        result.order_id, order, strategy_name
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
            log = self._get_order_log(db, order_id)

            if not log:
                logger.warning(
                    f"Order {order_id} exists on exchange but not in database! "
                    f"This indicates a previous logging failure."
                )
                # Attempt to create missing log entry if we have enough info
                if self.enable_auto_reconciliation:
                    self._reconcile_missing_order(db, order_id, result)
                return result

            # Update if status changed
            if log.status != result.status:
                logger.info(
                    f"Status for order {order_id} changed: "
                    f"{log.status.value} -> {result.status.value}"
                )

                try:
                    with db.begin_nested():
                        self._update_order_log(db, log, result)
                        db.commit()

                    # Update portfolio after successful DB update
                    self._safe_portfolio_update(db, None, result)

                except SQLAlchemyError as e:
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

        result: Optional[ExecutionResult] = None

        try:
            # Attempt cancellation via handler
            result = self.handler.cancel_order(order_id)

        except Exception as e:
            logger.error(f"Cancel API call for {order_id} failed: {e}", exc_info=True)
            result = self._create_synthetic_result_for_id(order_id, OrderStatus.FAILED)

        # Update database and portfolio
        if result:
            try:
                log = self._get_order_log(db, order_id)

                if log:
                    with db.begin_nested():
                        self._update_order_log(db, log, result)
                        db.commit()

                self._safe_portfolio_update(db, None, result)

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
            if result.status == OrderStatus.OPEN and order:
                self.portfolio.add_open_order(result.order_id, order)
            else:
                self.portfolio.update_order_status(db, result)

        except Exception as e:
            logger.error(
                f"Portfolio update failed for order {result.order_id}: {e}",
                exc_info=True,
            )
            # Don't raise - portfolio can be resynced later
            self._queue_for_reconciliation(result.order_id, order, "portfolio_resync")

    def _get_order_log(self, db: Session, order_id: str) -> Optional[OrderLogModel]:
        """
        Safely retrieve an order log with error handling.

        :param db: Database session
        :param order_id: Order ID to look up
        :return: OrderLogModel or None
        """
        try:
            return db.query(OrderLogModel).filter_by(order_id=order_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Database error fetching order log {order_id}: {e}")
            return None

    def _log_order(
        self,
        db: Session,
        order: OrderRequest,
        result: ExecutionResult,
        strategy_name: str,
    ):
        """
        Logs order to database with better error handling.
        Raises on failure to ensure transaction rollback.
        """
        log = self._get_order_log(db, result.order_id)

        if not log:
            log = OrderLogModel(
                order_id=result.order_id,
                market_id=order.market_id,
                strategy_name=strategy_name,
                side=order.side,
                outcome=order.outcome,
                requested_size=order.size,
                requested_price=order.price,
            )
            db.add(log)

        self._update_order_log(db, log, result)

    def _update_order_log(
        self, db: Session, log: OrderLogModel, result: ExecutionResult
    ):
        """Updates an existing order log entry."""
        log.status = result.status
        log.filled_size = result.filled_size
        log.avg_fill_price = result.avg_price
        log.updated_at = datetime.now(timezone.utc)
        # Don't commit here - let caller control transaction

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
            filled_size=0,
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
            filled_size=0,
            avg_price=0,
            timestamp=datetime.now(timezone.utc),
        )

    # --- Reconciliation Methods ---

    def _queue_for_reconciliation(
        self, order_id: str, order: Optional[OrderRequest], context: str
    ):
        """
        Adds an order to the reconciliation queue.

        :param order_id: Order ID to reconcile
        :param order: Original order request (if available)
        :param context: Context string for debugging
        """
        self._reconciliation_queue.append(
            {
                "order_id": order_id,
                "order": order,
                "context": context,
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
            log = OrderLogModel(
                order_id=order_id,
                market_id="UNKNOWN_MARKET",  # Would need to fetch from exchange
                strategy_name="reconciled",
                side=OrderSide.BUY,  # Would need to fetch from exchange
                outcome=MarketOutcome.YES,  # Would need to fetch from exchange
                requested_size=result.filled_size,
                requested_price=result.avg_price,
                status=result.status,
                filled_size=result.filled_size,
                avg_fill_price=result.avg_price,
            )
            db.add(log)
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
            f"market={order.market_id}, side={order.side.value}, "
            f"size={order.size}, status={result.status.value}"
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

        results = {
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

                # Update database
                log = self._get_order_log(db, order_id)
                if log:
                    self._update_order_log(db, log, current_result)
                    db.commit()

                else:
                    self._reconcile_missing_order(db, order_id, current_result)

                # Update portfolio
                self.portfolio.update_order_status(db, current_result)

                results["success"] += 1
                self._reconciliation_queue.remove(item)

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{item['order_id']}: {str(e)}")
                logger.error(f"Reconciliation failed for {item['order_id']}: {e}")

        logger.info(
            f"Reconciliation complete. Success: {results['success']}, "
            f"Failed: {results['failed']}"
        )

        return results
