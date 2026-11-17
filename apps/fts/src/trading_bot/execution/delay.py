# src/trading_bot/execution/delay.py

from .abc import ExecuteDelay


class KBarExecuteDelay(ExecuteDelay):
    """
    Delays execution by a fixed number of bars/ticks (T+k shift).
    """

    def __init__(self, k: int = 1) -> None:
        """
        Initializes the KBar delay model.

        :param k: The number of bars/ticks to delay. Must be >= 1.
        """
        if k < 1:
            raise ValueError(f"Execution delay shift k must be >= 1, got {k}")
        self.k = k

    def calculate_execution_tick(self, current_tick_index: int) -> int:
        return current_tick_index + self.k
