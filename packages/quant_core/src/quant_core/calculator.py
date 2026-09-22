from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize


class PortfolioCalculator:
    @staticmethod
    def cagr(beginning_value: float, ending_value: float, years: float) -> float:
        if beginning_value <= 0 or years <= 0:
            return 0.0
        return float((ending_value / beginning_value) ** (1 / years) - 1)

    @staticmethod
    def twr(period_returns: List[float]) -> float:
        # Time-Weighted Return
        twr = 1.0
        for r in period_returns:
            twr *= 1 + r
        return float(twr - 1)

    @staticmethod
    def xirr(cash_flows: List[Tuple[pd.Timestamp, float]]) -> float:
        # Internal Rate of Return (Money-Weighted Return)
        if not cash_flows:
            return 0.0

        df = pd.DataFrame(cash_flows, columns=["date", "amount"])
        df = df.sort_values("date")

        dates = df["date"].values
        amounts = df["amount"].values

        t0 = dates[0]
        days = np.array([(d - t0).astype("timedelta64[D]").astype(int) for d in dates])
        years = days / 365.0

        def npv(rate: float) -> float:
            if rate <= -1.0:
                return float("inf")
            return float(np.sum(amounts / ((1 + rate) ** years)))

        try:
            result = optimize.newton(npv, 0.1)
            return float(result)
        except (RuntimeError, ValueError):
            return 0.0

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
    ) -> float:
        if returns.empty or returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / periods_per_year
        return float(
            np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
        )

    @staticmethod
    def max_drawdown(values: pd.Series) -> float:
        if values.empty:
            return 0.0
        roll_max = values.cummax()
        drawdowns = values / roll_max - 1.0
        return float(drawdowns.min())
