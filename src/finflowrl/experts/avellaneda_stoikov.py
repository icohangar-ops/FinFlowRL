"""Avellaneda-Stoikov market-making expert policy.

Reference: Avellaneda & Stoikov (2008) "High-frequency trading in a
limit order book".

Quotes symmetric half-spread around a reservation price that accounts
for inventory risk:
    r = S + q * gamma * sigma^2 * (T - t)
    delta = gamma * sigma^2 * (T - t) + (1/gamma) * ln(1 + gamma/k)
"""

import numpy as np


class AvellanedaStoikovExpert:
    """Avellaneda-Stoikov market-making strategy."""

    def __init__(
        self,
        gamma: float = 0.1,     # risk aversion
        sigma: float = 0.02,    # volatility
        T: float = 60.0,        # time horizon
        k: float = 1.5,         # order arrival rate factor
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.T = T
        self.k = k

    def get_reservation_price(
        self, mid_price: float, inventory: float, t: float
    ) -> float:
        """Compute reservation price.

        Args:
            mid_price: current mid-price
            inventory: current inventory (positive = long)
            t: current time within [0, T]
        """
        tau = max(self.T - t, 1e-8)
        return mid_price - inventory * self.gamma * self.sigma ** 2 * tau

    def get_spread(self, t: float) -> float:
        """Compute optimal half-spread."""
        tau = max(self.T - t, 1e-8)
        return (
            self.gamma * self.sigma ** 2 * tau
            + (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)
        )

    def act(self, mid_price: float, inventory: float, t: float) -> dict:
        """Return bid/ask quotes.

        Returns:
            dict with bid_price, ask_price, half_spread
        """
        r = self.get_reservation_price(mid_price, inventory, t)
        half_spread = self.get_spread(t)
        return {
            "bid_price": r - half_spread,
            "ask_price": r + half_spread,
            "half_spread": half_spread,
            "reservation_price": r,
        }
