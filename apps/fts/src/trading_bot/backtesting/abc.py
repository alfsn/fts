# src/trading_bot/backtesting/abc.py

from abc import ABC, abstractmethod
from typing import Iterator

from ..core.schemas import IngestionEngineOutput


class BaseBacktestDataReader(ABC):
    """
    Abstract Base Class representing a data provider for historical simulation.
    Decoupled using Dependency Inversion, allowing the Replay Loop to drive
    backtesting regardless of data format (CSV, Parquet, SQLite).
    """

    @abstractmethod
    def read_data(self) -> Iterator[IngestionEngineOutput]:
        """
        Reads historical price or bar data sequentially and yields
        chronologically ordered IngestionEngineOutput packets.

        :return: An Iterator of IngestionEngineOutput schema contracts.
        """
        pass
