from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional, Sequence, Union

import pandas as pd

from .models import BarData, MarketDetails, OrderBook, Trade


class MarketDataProvider(ABC):
    @abstractmethod
    def get_bars(self, market_id: str, count: int = 100) -> Sequence[BarData]:
        pass

    @abstractmethod
    def get_order_book(self, market_id: str) -> Optional[OrderBook]:
        pass

    @abstractmethod
    def get_trade_history(self, market_id: str) -> Optional[Sequence[Trade]]:
        pass


class MarketDataRepository(ABC):
    @abstractmethod
    def get_prices(
        self,
        ticker: str,
        start_date: Union[date, datetime],
        end_date: Union[date, datetime],
    ) -> pd.Series:
        pass
