# src/trading_bot/core/repository.py

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from .enums import OrderSide, OrderStatus, PositionStatus
from .models import OrderLog as OrderLogModel
from .models import Position as PositionModel
from .schemas import Position as PositionSchema

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository holding the SQLAlchemy session."""

    def __init__(self, db: Session) -> None:
        self.db = db


class PositionRepository(BaseRepository):
    """Encapsulates all database operations for Position logs."""

    def get_open_positions(self) -> List[PositionSchema]:
        """Loads all open positions from the database."""
        try:
            open_models = (
                self.db.query(PositionModel).filter_by(status=PositionStatus.OPEN).all()
            )
            return [
                PositionSchema(
                    market_id=m.market_id,
                    outcome=m.outcome,
                    size=m.size,
                    entry_price=m.entry_price,
                )
                for m in open_models
            ]
        except Exception as e:
            logger.error(f"Failed to load open positions: {e}")
            return []

    def save_position(
        self, pos_schema: PositionSchema, is_delete: bool = False
    ) -> None:
        """
        Persists a Position schema state (Create, Update, or Close).
        """
        try:
            # Query by market_id and outcome (prediction markets might have multiple positions in one market_id)
            query = self.db.query(PositionModel).filter_by(
                market_id=pos_schema.market_id
            )
            if pos_schema.outcome:
                query = query.filter_by(outcome=pos_schema.outcome)
            else:
                query = query.filter(PositionModel.outcome.is_(None))

            pos_model = query.first()

            if is_delete:
                if pos_model:
                    pos_model.status = PositionStatus.CLOSED
                    pos_model.size = 0.0
            else:
                if not pos_model:
                    pos_model = PositionModel(
                        market_id=pos_schema.market_id,
                        outcome=pos_schema.outcome,
                        size=pos_schema.size,
                        entry_price=pos_schema.entry_price,
                        status=PositionStatus.OPEN,
                    )
                    self.db.add(pos_model)
                else:
                    pos_model.size = pos_schema.size
                    pos_model.entry_price = pos_schema.entry_price
                    pos_model.status = PositionStatus.OPEN

            self.db.commit()
            logger.debug(
                f"Position persisted for {pos_schema.market_id} (outcome: {pos_schema.outcome})"
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to persist position for {pos_schema.market_id}: {e}")
            raise e


class OrderRepository(BaseRepository):
    """Encapsulates all database operations for Order logs."""

    def create_order(
        self,
        order_id: str,
        market_id: str,
        strategy_name: Optional[str],
        side: OrderSide,
        outcome: Optional[str],
        requested_size: float,
        requested_price: float,
        status: OrderStatus = OrderStatus.PENDING,
    ) -> None:
        """Logs a new order request to the database."""
        try:
            order_model = OrderLogModel(
                order_id=order_id,
                market_id=market_id,
                strategy_name=strategy_name,
                side=side,
                outcome=outcome,
                requested_size=requested_size,
                requested_price=requested_price,
                status=status,
            )
            self.db.add(order_model)
            self.db.commit()
            logger.debug(f"Logged new order {order_id} in DB.")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to log order {order_id}: {e}")
            raise e

    def update_order(
        self,
        order_id: str,
        status: OrderStatus,
        filled_size: float,
        avg_fill_price: float,
    ) -> None:
        """Updates the status and fills details of an existing order."""
        try:
            order_model = (
                self.db.query(OrderLogModel).filter_by(order_id=order_id).first()
            )
            if order_model:
                order_model.status = status
                order_model.filled_size = filled_size
                order_model.avg_fill_price = avg_fill_price
                self.db.commit()
                logger.debug(f"Updated order {order_id} status to {status} in DB.")
            else:
                logger.warning(f"Order {order_id} not found in DB for status update.")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update order {order_id}: {e}")
            raise e
