"""HFT Gym-style Environment for market-making agents.

Implements an OpenAI Gym-compatible interface wrapping the MarketSimulator.
Observation: [inventory, mid_price, spread, volatility, order_imbalance, hawkes_intensity]
Action: continuous target_position in [-max_position, +max_position]
Reward: PnL change - inventory_risk_penalty - transaction_cost
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class HFTEnv:
    """High-Frequency Trading environment."""

    def __init__(
        self,
        max_steps: int = 1000,
        max_position: int = 10,
        transaction_cost: float = 0.001,
        inventory_penalty: float = 0.01,
        reward_scale: float = 1.0,
        seed: int = 42,
        obs_dim: int = 6,
        act_dim: int = 1,
    ):
        self.max_steps = max_steps
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.inventory_penalty = inventory_penalty
        self.reward_scale = reward_scale
        self.seed = seed

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.mid_price = 100.0
        self.prev_mid_price = 100.0
        self.total_pnl = 0.0
        self.done = False

        # Lazy import to avoid circular
        from ..simulator.market import MarketSimulator
        self.sim = MarketSimulator(seed=seed)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state. Returns observation."""
        self.current_step = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        self.done = False
        self.sim.reset()

        state = self.sim.step()
        self.mid_price = state["mid_price"]
        self.prev_mid_price = self.mid_price

        return self._get_obs(state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: target position (continuous, clipped to [-max_position, max_position])

        Returns:
            (observation, reward, done, info)
        """
        if self.done:
            return self.reset(), 0.0, True, {"pnl": self.total_pnl}

        target_pos = float(np.clip(action[0] if np.ndim(action) > 0 else action,
                                    -self.max_position, self.max_position))

        # Get market state
        state = self.sim.step()
        self.prev_mid_price = self.mid_price
        self.mid_price = state["mid_price"]

        # Execute trade to reach target position
        trade_qty = target_pos - self.inventory
        trade_cost = abs(trade_qty) * self.transaction_cost
        self.inventory = target_pos
        self.cash -= trade_cost

        # Mark-to-market PnL
        unrealized_pnl = self.inventory * (self.mid_price - self.prev_mid_price)
        realized_pnl = -trade_cost
        step_pnl = unrealized_pnl + realized_pnl

        # Inventory risk penalty (quadratic)
        inv_penalty = self.inventory_penalty * (self.inventory / self.max_position) ** 2

        reward = (step_pnl - inv_penalty) * self.reward_scale
        self.total_pnl += step_pnl

        self.current_step += 1
        self.done = self.current_step >= self.max_steps

        obs = self._get_obs(state)
        info = {
            "pnl": self.total_pnl,
            "step_pnl": step_pnl,
            "inventory": self.inventory,
            "mid_price": self.mid_price,
            "trade_qty": trade_qty,
        }

        return obs, reward, self.done, info

    def _get_obs(self, state: dict) -> np.ndarray:
        """Construct observation vector."""
        return np.array([
            self.inventory / max(self.max_position, 1.0),
            (self.mid_price - self.prev_mid_price) / self.mid_price,
            state["spread"],
            state.get("spread", 0.02) * 10,  # proxy volatility
            state.get("inventory_shock", 0.0) / 10.0,
            state.get("hawkes_intensity", 5.0) / 20.0,
        ], dtype=np.float64)
