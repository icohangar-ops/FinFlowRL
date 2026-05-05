"""
Market Simulator — Jump-diffusion process + Hawkes process for order flow.

Models realistic market microstructure:
  - Mid-price follows Merton jump-diffusion
  - Order arrivals follow a self-exciting Hawkes process
  - Supports configurable spread, volatility, and intensity parameters
"""

import numpy as np
from typing import Optional, Dict, Any


class MarketSimulator:
    """Simulates a limit-order-book market environment.

    The mid-price evolves via a Merton jump-diffusion:
        dS = mu*dt + sigma*dW + J*dN

    Order arrivals follow a Hawkes process with exponential kernel:
        lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
    """

    def __init__(
        self,
        seed: int = 42,
        S0: float = 100.0,
        mu: float = 0.0,
        sigma: float = 0.02,
        jump_intensity: float = 0.1,
        jump_mean: float = 0.0,
        jump_std: float = 0.01,
        half_spread: float = 0.01,
        hawkes_mu: float = 5.0,
        hawkes_alpha: float = 2.0,
        hawkes_beta: float = 10.0,
        dt: float = 1.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.half_spread = half_spread
        self.hawkes_mu = hawkes_mu
        self.hawkes_alpha = hawkes_alpha
        self.hawkes_beta = hawkes_beta
        self.dt = dt

        self.mid_price = S0
        self.order_history: list = []
        self.hawkes_intensity = hawkes_mu

    def reset(self, S0: Optional[float] = None) -> float:
        """Reset the simulator to initial state. Returns starting mid-price."""
        if S0 is not None:
            self.S0 = S0
        self.mid_price = self.S0
        self.order_history.clear()
        self.hawkes_intensity = self.hawkes_mu
        return self.mid_price

    def step(self) -> Dict[str, Any]:
        """Advance one time step. Returns market state dict.

        Returns:
            dict with keys: mid_price, best_bid, best_ask, spread,
            order_arrivals, hawkes_intensity, inventory_shock
        """
        # --- Mid-price dynamics (jump-diffusion) ---
        dW = self.rng.normal(0, np.sqrt(self.dt))
        jump = 0.0
        n_jumps = self.rng.poisson(self.jump_intensity * self.dt)
        if n_jumps > 0:
            jump = self.rng.normal(self.jump_mean, self.jump_std, n_jumps).sum()

        dS = self.mu * self.dt + self.sigma * dW + jump
        self.mid_price += dS
        self.mid_price = max(self.mid_price, 1e-6)  # floor

        # --- Hawkes order arrivals ---
        n_orders = self.rng.poisson(max(self.hawkes_intensity * self.dt, 0))
        # Update intensity: self-exciting decay + re-seed
        self.hawkes_intensity = (
            self.hawkes_mu
            + self.hawkes_alpha * n_orders
        ) * np.exp(-self.hawkes_beta * self.dt)
        self.hawkes_intensity = max(self.hawkes_intensity, self.hawkes_mu * 0.5)

        # Random walk on half-spread for realism
        spread_noise = self.rng.normal(0, 0.001)
        effective_half_spread = max(self.half_spread + spread_noise, 1e-5)

        best_bid = self.mid_price - effective_half_spread
        best_ask = self.mid_price + effective_half_spread

        # Inventory shock: net order flow imbalance
        buy_orders = self.rng.binomial(n_orders, 0.5)
        sell_orders = n_orders - buy_orders
        inventory_shock = buy_orders - sell_orders

        self.order_history.append({
            "t": len(self.order_history),
            "n_orders": n_orders,
            "buy": buy_orders,
            "sell": sell_orders,
        })

        return {
            "mid_price": self.mid_price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": best_ask - best_bid,
            "order_arrivals": n_orders,
            "hawkes_intensity": self.hawkes_intensity,
            "inventory_shock": inventory_shock,
        }

    def simulate(self, n_steps: int) -> Dict[str, np.ndarray]:
        """Run simulation for n_steps, return arrays of market data.

        Returns:
            dict with numpy arrays: mid_prices, best_bids, best_asks,
            spreads, order_arrivals, hawkes_intensities, inventory_shocks
        """
        self.reset()
        history = {k: [] for k in [
            "mid_price", "best_bid", "best_ask", "spread",
            "order_arrivals", "hawkes_intensity", "inventory_shock",
        ]}
        for _ in range(n_steps):
            state = self.step()
            for k, v in state.items():
                history[k].append(v)
        return {k: np.array(v) for k, v in history.items()}
