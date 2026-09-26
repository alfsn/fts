from datetime import datetime
from typing import List, Optional

from quant_core.enums import BarType
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

    market: Mapped["Market"] = relationship("Market", back_populates="bars")
