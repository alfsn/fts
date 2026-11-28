from datetime import datetime
from typing import List, Optional

from quant_core.enums import BarType, CorporateActionType
from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    instrument_id: Mapped[str] = mapped_column(
        String, unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    resolution_source: Mapped[Optional[str]] = mapped_column(String)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    bars: Mapped[List["BarDataLog"]] = relationship(
        "BarDataLog", back_populates="market"
    )


class BarDataLog(Base):
    __tablename__ = "bar_data_logs"

    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "timestamp", "bar_type", "interval", name="uq_bar_data_log"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    instrument_id: Mapped[str] = mapped_column(
        String, ForeignKey("markets.instrument_id"), index=True, nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )

    open: Mapped[float] = mapped_column(nullable=False)
    high: Mapped[float] = mapped_column(nullable=False)
    low: Mapped[float] = mapped_column(nullable=False)
    close: Mapped[float] = mapped_column(nullable=False)
    volume: Mapped[float] = mapped_column(nullable=False)

    bar_type: Mapped[BarType] = mapped_column(Enum(BarType), nullable=False, index=True)
    interval: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    ticks_count: Mapped[int] = mapped_column(nullable=False)
    dollar_volume: Mapped[float] = mapped_column(nullable=False)
    # Vendor that produced the bar (provider.source_name). One source per
    # symbol and interval: a series never mixes vendors' conventions (PR 2a, D4).
    source: Mapped[str] = mapped_column(String, nullable=False)

    market: Mapped["Market"] = relationship("Market", back_populates="bars")


class CorporateActionRecord(Base):
    __tablename__ = "corporate_actions"

    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "action_type", "ex_date", name="uq_corporate_action"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    instrument_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    action_type: Mapped[CorporateActionType] = mapped_column(
        Enum(CorporateActionType), index=True, nullable=False
    )
    ex_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )
    record_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    payable_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    amount: Mapped[Optional[float]] = mapped_column(nullable=True)
    ratio: Mapped[Optional[float]] = mapped_column(nullable=True)
    # Same vendor as the symbol's bars, so splits and prices agree (PR 2a, D8).
    source: Mapped[str] = mapped_column(String, nullable=False)
