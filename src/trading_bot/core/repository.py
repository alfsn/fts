# src/trading_bot/core/repository.py

import logging
from datetime import datetime
from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from .enums import BarType, OrderSide, OrderStatus, PositionStatus
from .models import BarDataLog as BarDataLogModel
from .models import Market as MarketModel
from .models import ModelRegistryLog
from .models import OrderLog as OrderLogModel
from .models import Position as PositionModel
from .models import TimeSeriesDataset
from .schemas import BarData as BarDataSchema
from .schemas import MarketDetails as MarketDetailsSchema
from .schemas import Position as PositionSchema

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository holding the SQLAlchemy session."""

    def __init__(self, db: Session) -> None:
        self.db = db


class PositionRepository(BaseRepository):
    """Encapsulates all database operations for Position logs."""

    def get_open_positions(self, run_id: Optional[str] = None) -> List[PositionSchema]:
        """Loads all open positions from the database."""
        try:
            query = self.db.query(PositionModel).filter_by(status=PositionStatus.OPEN)
            if run_id:
                query = query.filter_by(run_id=run_id)
            open_models = query.all()
            return [
                PositionSchema(
                    market_id=m.market_id,
                    outcome=m.outcome,
                    size=m.size,
                    entry_price=m.entry_price,
                    run_id=m.run_id,
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
            # Query by market_id, outcome, and run_id
            query = self.db.query(PositionModel).filter_by(
                market_id=pos_schema.market_id,
                run_id=pos_schema.run_id,
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
                        run_id=pos_schema.run_id,
                        size=pos_schema.size,
                        entry_price=pos_schema.entry_price,
                        status=PositionStatus.OPEN,
                    )
                    self.db.add(pos_model)
                else:
                    pos_model.size = pos_schema.size
                    pos_model.entry_price = pos_schema.entry_price
                    pos_model.status = PositionStatus.OPEN

            logger.debug(
                f"Position persisted for {pos_schema.market_id} (outcome: {pos_schema.outcome}, run_id: {pos_schema.run_id})"
            )
        except Exception as e:
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
            logger.debug(f"Logged new order {order_id} in DB.")
        except Exception as e:
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
                logger.debug(f"Updated order {order_id} status to {status} in DB.")
            else:
                logger.warning(f"Order {order_id} not found in DB for status update.")
        except Exception as e:
            logger.error(f"Failed to update order {order_id}: {e}")
            raise e


class MarketDataRepository(BaseRepository):
    """Encapsulates all database operations for market and bar data."""

    def ensure_market(self, details: MarketDetailsSchema) -> MarketModel:
        """Ensures that the market exists in the database, updating details if needed."""
        try:
            market = (
                self.db.query(MarketModel)
                .filter_by(market_id=details.market_id)
                .first()
            )
            if not market:
                market = MarketModel(
                    market_id=details.market_id,
                    name=details.name,
                    end_date=details.end_date,
                    resolution_source=details.resolution_source,
                )
                self.db.add(market)
            else:
                market.name = details.name
                market.end_date = details.end_date
                market.resolution_source = details.resolution_source
            return market
        except Exception as e:
            logger.error(f"Failed to ensure market {details.market_id}: {e}")
            raise e

    def save_bars(self, market_id: str, bars: Sequence[BarDataSchema]) -> int:
        """
        Saves a sequence of BarData schema elements to the database.
        Avoids duplicates based on market_id, timestamp, bar_type, and interval.
        Returns the number of new bars inserted.
        """
        if not bars:
            return 0

        try:
            bar_type = bars[0].bar_type
            interval = bars[0].interval
            from datetime import timezone

            # Load existing timestamps to prevent duplicate insertions
            existing_records = (
                self.db.query(BarDataLogModel.timestamp)
                .filter_by(market_id=market_id, bar_type=bar_type, interval=interval)
                .all()
            )
            # Normalize DB naive datetimes to naive UTC (assuming they were saved in UTC)
            timestamps = {
                (
                    r[0].astimezone(timezone.utc).replace(tzinfo=None)
                    if r[0].tzinfo
                    else r[0]
                )
                for r in existing_records
            }

            new_logs = []
            for bar in bars:
                ts = bar.timestamp
                ts_normalized = (
                    ts.astimezone(timezone.utc).replace(tzinfo=None)
                    if ts.tzinfo
                    else ts
                )
                if ts_normalized not in timestamps:
                    log = BarDataLogModel(
                        market_id=market_id,
                        timestamp=ts_normalized,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        bar_type=bar.bar_type,
                        interval=bar.interval,
                        ticks_count=bar.ticks_count,
                        dollar_volume=bar.dollar_volume,
                    )
                    new_logs.append(log)
                    timestamps.add(ts_normalized)

            if new_logs:
                self.db.bulk_save_objects(new_logs)
                logger.info(f"Saved {len(new_logs)} new bars for {market_id} to SQL.")
            return len(new_logs)
        except Exception as e:
            logger.error(f"Failed to save bars for {market_id}: {e}")
            raise e

    def get_bars(
        self,
        market_ids: str | Sequence[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bar_type: BarType = BarType.TIME,
        interval: Optional[str] = None,
    ) -> List[BarDataLogModel]:
        """
        Loads historical bars for given market(s) and date range, ordered by timestamp ascending.

        :param market_ids: Market ID string or sequence of market ID strings.
        :param start_date: Optional start datetime (inclusive).
        :param end_date: Optional end datetime (inclusive).
        :param bar_type: The type of bar to query (default: BarType.TIME).
        :param interval: Optional timeframe/interval filter (e.g., '1m', '1h').
        :return: A list of BarDataLog DB models.
        """
        try:
            if isinstance(market_ids, str):
                market_ids = [market_ids]

            query = self.db.query(BarDataLogModel).filter(
                BarDataLogModel.market_id.in_(market_ids),
                BarDataLogModel.bar_type == bar_type,
            )
            if interval is not None:
                query = query.filter(BarDataLogModel.interval == interval)
            if start_date:
                query = query.filter(BarDataLogModel.timestamp >= start_date)
            if end_date:
                query = query.filter(BarDataLogModel.timestamp <= end_date)

            return query.order_by(BarDataLogModel.timestamp.asc()).all()
        except Exception as e:
            logger.error(f"Failed to load bars for {market_ids}: {e}")
            raise e


class ModelRepository(BaseRepository):
    """Encapsulates database operations for the Model Registry. Decoupled from transaction control."""

    def get_or_create_dataset(
        self,
        market_id: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        hash_val: str,
    ) -> TimeSeriesDataset:
        """Retrieves an existing dataset by its content hash or creates a new entry."""
        existing = self.db.query(TimeSeriesDataset).filter_by(hash=hash_val).first()
        if existing:
            return existing

        dataset_id = f"ds_{hash_val[:12]}"
        dataset = TimeSeriesDataset(
            dataset_id=dataset_id,
            market_id=market_id,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            hash=hash_val,
        )
        self.db.add(dataset)
        return dataset

    def register_model(
        self,
        model_id: str,
        model_type: str,
        market_id: str,
        interval: str,
        horizon: int,
        onnx_path: str,
        hyperparameters: dict,
        metrics: dict,
        run_id: Optional[str] = None,
        status: str = "candidate",
        dataset_id: Optional[str] = None,
    ) -> ModelRegistryLog:
        """Adds a trained model metadata entry to the registry session idempotently."""
        existing = self.db.query(ModelRegistryLog).filter_by(model_id=model_id).first()
        if existing:
            logger.info(
                f"Model '{model_id}' already registered in model_registry. "
                "Returning existing entry."
            )
            return existing

        log_entry = ModelRegistryLog(
            model_id=model_id,
            run_id=run_id,
            model_type=model_type,
            market_id=market_id,
            interval=interval,
            horizon=horizon,
            onnx_path=onnx_path,
            hyperparameters=hyperparameters,
            metrics=metrics,
            status=status,
            dataset_id=dataset_id,
        )
        self.db.add(log_entry)
        return log_entry

    def get_production_model(
        self, model_type: str, market_id: str, interval: str, horizon: int
    ) -> Optional[ModelRegistryLog]:
        """Fetches the active 'production' model matching the logical signature."""
        return (
            self.db.query(ModelRegistryLog)
            .filter_by(
                model_type=model_type,
                market_id=market_id,
                interval=interval,
                horizon=horizon,
                status="production",
            )
            .order_by(ModelRegistryLog.created_at.desc())
            .first()
        )

    def get_model(self, model_id: str) -> Optional[ModelRegistryLog]:
        """Fetches a model by its unique model ID."""
        return self.db.query(ModelRegistryLog).filter_by(model_id=model_id).first()

    def promote_to_production(self, model_id: str) -> None:
        """Promotes a candidate model, moves its ONNX file to permanent registry,
        archives previous production models, and cleans up other candidate files/records.
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model with model_id {model_id} not found.")

        import os
        import shutil

        # Move ONNX file from trials/ to permanent registry if applicable
        source_path = model.onnx_path
        if source_path and "registry/trials" in source_path:
            # permanent registry is the parent directory of registry/trials/
            registry_dir = os.path.dirname(os.path.dirname(source_path))
            permanent_path = os.path.join(registry_dir, f"{model_id}.onnx")

            if os.path.exists(source_path):
                shutil.move(source_path, permanent_path)
                model.onnx_path = permanent_path
            else:
                logger.warning(
                    f"ONNX file not found at expected trial path: {source_path}"
                )

        # Demote current production model(s) with matching logical signature
        self.db.query(ModelRegistryLog).filter_by(
            model_type=model.model_type,
            market_id=model.market_id,
            interval=model.interval,
            horizon=model.horizon,
            status="production",
        ).update({"status": "archived"})

        # Promote this model
        model.status = "production"

        # Clean up files of other candidates with the same logical signature
        other_candidates = (
            self.db.query(ModelRegistryLog)
            .filter(
                ModelRegistryLog.model_type == model.model_type,
                ModelRegistryLog.market_id == model.market_id,
                ModelRegistryLog.interval == model.interval,
                ModelRegistryLog.horizon == model.horizon,
                ModelRegistryLog.model_id != model_id,
                ModelRegistryLog.status == "candidate",
            )
            .all()
        )

        for candidate in other_candidates:
            if candidate.onnx_path and os.path.exists(candidate.onnx_path):
                try:
                    os.remove(candidate.onnx_path)
                except Exception as e:
                    logger.warning(
                        f"Failed to remove unpromoted candidate file {candidate.onnx_path}: {e}"
                    )
            self.db.delete(candidate)
