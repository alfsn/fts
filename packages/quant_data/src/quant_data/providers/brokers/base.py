from abc import ABC, abstractmethod

from quant_core.models import Portfolio


class BrokerConnector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_portfolio(self) -> Portfolio:
        pass
