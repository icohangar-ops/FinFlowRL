"""Synthetic market data generation.

Thin wrapper around :class:`~finflowrl.simulator.market.MarketSimulator`
that runs the jump-diffusion + Hawkes simulation and optionally persists the
resulting time series to a compressed ``.npz`` file.
"""

from typing import Dict, Optional

import numpy as np

from ..simulator.market import MarketSimulator


def generate_market_data(
    n_steps: int = 10000,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Generate synthetic market data.

    Args:
        n_steps: Number of simulation steps to run.
        seed: Random seed for reproducibility.
        save_path: Optional path to save the data as a compressed ``.npz``.

    Returns:
        Dict of numpy arrays with keys: ``mid_price``, ``best_bid``,
        ``best_ask``, ``spread``, ``order_arrivals``, ``hawkes_intensity``,
        ``inventory_shock``.
    """
    sim = MarketSimulator(seed=seed)
    data = sim.simulate(n_steps=n_steps)
    if save_path is not None:
        np.savez_compressed(save_path, **data)
    return data
