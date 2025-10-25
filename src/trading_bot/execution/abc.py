"""
Abstract Base Classes for the Execution Engine (Module 4).

This file defines the abstract interface for an Execution Handler,
which is responsible for the technical implementation of placing,
canceling, and monitoring orders on a specific exchange.
"""

from abc import ABC, abstractmethod
from typing import Dict

from ...core.enums import MarketName
from ...core.schemas import ExecutionResult, OrderRequest


class BaseExecutionHandler(ABC):
    """
    Abstract base class for an execution handler.

    Defines the interface for interacting with an exchange's
    trading functionality (placing, canceling, and checking orders).
    A concrete implementation will exist for each exchange (e.g.,
    PolymarketHandler) and will manage the private keys,
    transaction signing, and API communication.
    """

    @property
    @abstractmethod
    def market_name(self) -> MarketName:
        """
        The specific market this handler is built for (e.g.,
        MarketName.POLYMARKET).

        :return: A MarketName enum member.
        """
        pass

    @abstractmethod
    def execute_order(self, order: OrderRequest) -> ExecutionResult:
        """
        Submits a trade order to the exchange.

        This method must handle the entire lifecycle: building the
        transaction, signing it with the bot's private key,
        and sending it to the exchange's API or smart contract.
        It should return immediately with a pending status.

        :param order: The OrderRequest object detailing the trade.
        :return: An ExecutionResult object with the immediate
                 status (e.g., PENDING or FAILED) and order_id.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> ExecutionResult:
        """
        Attempts to cancel a previously placed open order.

        :param order_id: The unique identifier of the order to cancel
                         (as returned by execute_order).
        :return: An ExecutionResult object, ideally with the
                 status set to CANCELED or FAILED.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> ExecutionResult:
        """
        Queries the exchange for the current status of an order.

        This is used to update the portfolio and check if
        an order has been filled, partially filled, or rejected.

        :param order_id: The unique identifier of the order.
        :return: An ExecutionResult object populated with the
                 latest status (e.g., OPEN, FILLED, FAILED).
        """
        pass

    @abstractmethod
    def get_account_balances(self) -> Dict[str, float]:
        """
        Fetches the bot's wallet balances from the exchange/blockchain.

        This is crucial for the Risk Manager to know the total and
        available capital, as well as the gas token balance.

        :return: A dictionary containing key balances, e.g.,
                 {'USDC': 10000.0, 'MATIC': 10.5}
        """
        pass
