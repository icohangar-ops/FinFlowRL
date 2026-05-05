"""Evaluation Metrics — PnL, Sharpe Ratio, Maximum Drawdown."""

import numpy as np
from typing import List


def compute_pnl(returns: List[float]) -> float:
    """Compute cumulative PnL from a list of per-step returns.

    Args:
        returns: list of per-step PnL values

    Returns:
        cumulative PnL
    """
    return float(np.sum(returns))


def compute_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualised Sharpe ratio.

    Args:
        returns: list of per-step returns
        risk_free_rate: annualised risk-free rate
        periods_per_year: number of trading periods per year

    Returns:
        annualised Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    r = np.array(returns)
    excess = r - risk_free_rate / periods_per_year
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)
    if std_excess < 1e-10:
        return 0.0
    return float(mean_excess / std_excess * np.sqrt(periods_per_year))


def compute_max_drawdown(returns: List[float]) -> float:
    """Compute maximum drawdown from a list of per-step returns.

    Args:
        returns: list of per-step returns

    Returns:
        maximum drawdown (positive number, e.g. 0.15 = 15%)
    """
    if len(returns) < 2:
        return 0.0
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return float(np.max(drawdowns))
