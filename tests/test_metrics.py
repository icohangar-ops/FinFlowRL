"""Tests for Evaluation Metrics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from finflowrl.evaluation.metrics import compute_pnl, compute_sharpe_ratio, compute_max_drawdown


def test_pnl():
    returns = [1.0, -0.5, 2.0, -1.0]
    pnl = compute_pnl(returns)
    assert pnl == 1.5


def test_pnl_empty():
    pnl = compute_pnl([])
    assert pnl == 0.0


def test_sharpe():
    returns = [0.01, 0.02, 0.015, -0.005, 0.03, 0.025, 0.01, -0.01, 0.02, 0.015]
    sr = compute_sharpe_ratio(returns)
    assert sr > 0  # positive mean returns
    assert np.isfinite(sr)


def test_sharpe_short():
    sr = compute_sharpe_ratio([0.01])
    assert sr == 0.0  # not enough data


def test_max_drawdown():
    returns = [1.0, 2.0, -3.0, 1.0, 2.0]
    mdd = compute_max_drawdown(returns)
    assert mdd > 0
    assert np.isfinite(mdd)


def test_max_drawdown_monotonic():
    returns = [1.0, 2.0, 3.0, 4.0]
    mdd = compute_max_drawdown(returns)
    assert mdd == 0.0  # always going up


if __name__ == "__main__":
    test_pnl()
    test_pnl_empty()
    test_sharpe()
    test_sharpe_short()
    test_max_drawdown()
    test_max_drawdown_monotonic()
    print("All metrics tests passed!")
