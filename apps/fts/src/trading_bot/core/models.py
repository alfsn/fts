# src/trading_bot/core/models.py
"""fts tables that are not part of a run. The run tables live in
``trading_bot.runs.tables`` and only ``trading_bot.runs`` may import them."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base
from .enums import AlertSeverity


class EventLog(Base):
    """
    SQLAlchemy ORM Model for generic, text-based event logging.

    Used for monitoring, debugging, and persisting alerts.
    Maps to the Alert Pydantic schema.
    """

    __tablename__ = "event_logs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    # The severity of the event (e.g., INFO, ERROR)
    severity: Mapped[AlertSeverity] = mapped_column(
        Enum(AlertSeverity), index=True, nullable=False
    )
    # The module that generated the log (e.g., 'StrategyEngine')
    source_module: Mapped[Optional[str]] = mapped_column(String, index=True)
    # Use Text for long log messages
    message: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<EventLog(timestamp='{self.timestamp}', "
            f"severity='{self.severity}', message='{self.message[:50]}...')>"
        )


class TimeSeriesDataset(Base):
    """
    SQLAlchemy ORM Model representing a dataset.
    Identifies a specific versioned slice of time-series bar data.
    """

    __tablename__ = "time_series_datasets"

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    instrument_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    interval: Mapped[str] = mapped_column(String, index=True, nullable=False)
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hash: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)

    def __repr__(self) -> str:
        return f"<TimeSeriesDataset(dataset_id='{self.dataset_id}', instrument_id='{self.instrument_id}', interval='{self.interval}')>"


class ModelRegistryLog(Base):
    """
    SQLAlchemy ORM Model representing the model registry.
    Stores metadata, serialized paths, hyperparameters, and evaluation metrics.
    """

    __tablename__ = "model_registry"

    __table_args__ = (
        Index(
            "ix_model_registry_signature",
            "model_type",
            "instrument_id",
            "interval",
            "horizon",
            "status",
        ),
        Index(
            "uq_production_model_signature",
            "model_type",
            "instrument_id",
            "interval",
            "horizon",
            unique=True,
            sqlite_where=text("status = 'production'"),
        ),
    )

    model_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    run_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    model_type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    instrument_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    interval: Mapped[str] = mapped_column(String, index=True, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    dataset_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("time_series_datasets.dataset_id"), nullable=True, index=True
    )
    dataset: Mapped[Optional["TimeSeriesDataset"]] = relationship("TimeSeriesDataset")

    onnx_path: Mapped[str] = mapped_column(String, nullable=False)
    hyperparameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String, default="candidate", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<ModelRegistryLog(model_id='{self.model_id}', type='{self.model_type}', status='{self.status}')>"
