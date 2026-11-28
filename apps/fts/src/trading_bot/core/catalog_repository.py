# src/trading_bot/core/catalog_repository.py

"""
Repository and Query Service Layer for Quant Data Catalog & Backtest Visibility.

Provides SOLID-compliant abstractions for retrieving models, backtest runs,
computing performance metrics dynamically, and managing model promotion lifecycles.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from ..runs import RunRecord, RunStore
from .models import BarDataLog, Market, ModelRegistryLog
from .performance import (
    build_benchmark_metrics,
    build_performance_metrics,
    build_vol_forecast,
)
from .schemas import (
    BacktestDetailDTO,
    BacktestRunCatalogItem,
    ModelCatalogItem,
    ModelDetailDTO,
)


class ModelCatalogRepository:
    """
    Repository for querying ModelRegistryLog entries and managing status transitions.
    """

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def list_models(
        self,
        model_type: Optional[str] = None,
        instrument_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ModelCatalogItem]:
        """List and filter models from model_registry."""
        with self.session_factory() as session:
            stmt = select(ModelRegistryLog)

            if model_type:
                stmt = stmt.where(ModelRegistryLog.model_type == model_type)
            if instrument_id:
                stmt = stmt.where(ModelRegistryLog.instrument_id == instrument_id)
            if status:
                stmt = stmt.where(ModelRegistryLog.status == status)

            stmt = stmt.order_by(ModelRegistryLog.created_at.desc())
            records = session.scalars(stmt).all()

            items = [ModelCatalogItem.model_validate(r) for r in records]
            return items

    def get_model_details(self, model_id: str) -> Optional[ModelDetailDTO]:
        """Get detailed DTO for a specific model."""
        with self.session_factory() as session:
            stmt = select(ModelRegistryLog).where(ModelRegistryLog.model_id == model_id)
            record = session.scalar(stmt)
            if not record:
                return None

            onnx_exists = (
                os.path.exists(record.onnx_path) if record.onnx_path else False
            )

            dto = ModelDetailDTO.model_validate(record)
            dto.onnx_exists = onnx_exists
            return dto

    def update_model_status(self, model_id: str, new_status: str) -> bool:
        """
        Update model status. If new_status is 'production', demote any active production model
        matching the (model_type, instrument_id, interval, horizon) signature back to 'candidate'
        within an atomic transaction.
        """
        with self.session_factory() as session:
            with session.begin():
                model = session.scalar(
                    select(ModelRegistryLog).where(
                        ModelRegistryLog.model_id == model_id
                    )
                )
                if not model:
                    return False

                if new_status.lower() == "production":
                    # Demote existing production models matching signature to 'candidate'
                    stmt_demote = (
                        update(ModelRegistryLog)
                        .where(
                            ModelRegistryLog.model_type == model.model_type,
                            ModelRegistryLog.instrument_id == model.instrument_id,
                            ModelRegistryLog.interval == model.interval,
                            ModelRegistryLog.horizon == model.horizon,
                            ModelRegistryLog.status == "production",
                            ModelRegistryLog.model_id != model_id,
                        )
                        .values(
                            status="candidate", updated_at=datetime.now(timezone.utc)
                        )
                    )
                    session.execute(stmt_demote)

                model.status = new_status.lower()
                model.updated_at = datetime.now(timezone.utc)
            return True


class BacktestCatalogRepository:
    """
    Backtest runs with their KPIs. Runs are read through ``RunStore``; the
    linked model comes from the run's segment (M1), never from its name.
    """

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory
        self.store = RunStore(session_factory)

    def _model(self, record: RunRecord) -> Optional[ModelRegistryLog]:
        if record.model_id is None:
            return None
        with self.session_factory() as session:
            model = session.get(ModelRegistryLog, record.model_id)
            if model is not None:
                session.expunge(model)
            return model

    def list_runs(
        self, instrument_id: Optional[str] = None, min_sharpe: Optional[float] = None
    ) -> List[BacktestRunCatalogItem]:
        """List backtest runs with dynamically calculated summary metrics."""
        scope = (
            None
            if not instrument_id or instrument_id.upper() == "ALL"
            else instrument_id
        )
        items = []
        for header in self.store.list(kind="backtest", instrument_id=scope):
            record = self.store.load(header.run_id, predictions=False)
            if record is None or not record.equity:
                continue

            metrics = build_performance_metrics(record.equity, record.fills)

            # NaN Sharpe (too short to annualise) never satisfies a minimum.
            if min_sharpe is not None and not metrics.sharpe_ratio >= min_sharpe:
                continue

            model = self._model(record)
            items.append(
                BacktestRunCatalogItem(
                    run_id=header.run_id,
                    strategy_name=header.strategy_name or "unknown",
                    instrument_id=header.instrument_id or "ALL",
                    model_id=record.model_id,
                    hyperparameters=(
                        model.hyperparameters
                        if model is not None and isinstance(model.hyperparameters, dict)
                        else {}
                    ),
                    start_time=record.equity[0].timestamp,
                    end_time=record.equity[-1].timestamp,
                    metrics=metrics,
                    benchmark_metrics=build_benchmark_metrics(record.equity),
                )
            )
        return items

    def get_run_details(self, run_id: str) -> Optional[BacktestDetailDTO]:
        """Get time-series equity curves, fills and prediction logs for a run."""
        record = self.store.load(run_id)
        if record is None or not record.equity:
            return None

        return BacktestDetailDTO(
            run_id=run_id,
            strategy_name=record.header.strategy_name or "unknown",
            instrument_id=record.header.instrument_id or "ALL",
            start_time=record.equity[0].timestamp,
            end_time=record.equity[-1].timestamp,
            metrics=build_performance_metrics(record.equity, record.fills),
            benchmark_metrics=build_benchmark_metrics(record.equity),
            vol_forecast=build_vol_forecast(record.equity),
            equity_curve=[
                {**p.model_dump(), "timestamp": p.timestamp.isoformat()}
                for p in record.equity
            ],
            trades=[
                {
                    "order_id": f.order_id,
                    "instrument_id": f.instrument_id,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "fill_price": f.fill_price,
                    "fill_timestamp": f.fill_timestamp.isoformat(),
                }
                for f in record.fills
            ],
            predictions=[
                {
                    "timestamp": p.timestamp.isoformat(),
                    "instrument_id": p.instrument_id,
                    "predicted_signal": p.predicted_signal,
                    "confidence": p.confidence,
                    "actual_future_return": p.actual_future_return,
                }
                for p in record.predictions
            ],
        )


class CatalogQueryService:
    """
    Facade service exposing high-level data catalog query APIs and database health stats.
    """

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory
        self.model_repo = ModelCatalogRepository(session_factory)
        self.backtest_repo = BacktestCatalogRepository(session_factory)

    def get_database_summary(self) -> Dict[str, int]:
        """Retrieve total counts across core database tables."""

        def _count(session, model_cls):
            try:
                return session.scalar(select(func.count()).select_from(model_cls)) or 0
            except Exception:
                return 0

        with self.session_factory() as session:
            markets = _count(session, Market)
            models = _count(session, ModelRegistryLog)
            production_models = (
                session.scalar(
                    select(func.count())
                    .select_from(ModelRegistryLog)
                    .where(ModelRegistryLog.status == "production")
                )
                or 0
            )
            bar_logs = _count(session, BarDataLog)

        runs = self.backtest_repo.store.counts()
        return {
            "markets": markets,
            "bar_logs": bar_logs,
            "order_logs": runs["order_logs"],
            "trade_logs": runs["trade_logs"],
            "models": models,
            "production_models": production_models,
            "equity_log_ticks": runs["equity_log_ticks"],
            "prediction_logs": runs["prediction_logs"],
        }
