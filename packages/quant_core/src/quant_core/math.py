import numpy as np
import pandas as pd


def calculate_cagr(returns: pd.Series) -> float:
    """Calculate Compound Annual Growth Rate."""
    if len(returns) == 0:
        return 0.0
    cum_return = (1 + returns).prod()
    years = len(returns) / 252.0  # Assuming daily returns
    if years <= 0:
        return 0.0
    return float(cum_return ** (1 / years) - 1)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate Maximum Drawdown."""
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / peak) - 1
    return float(drawdown.min())


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Annualized Sharpe Ratio."""
    if len(returns) == 0:
        return 0.0
    mean_return = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    if volatility == 0:
        return 0.0
    return float(mean_return / volatility)
