"""GLFT (Generalized Linear Feature-based Trading) expert policy.

A linear-quadratic market-making strategy that extends Avellaneda-Stoikov
with additional feature inputs (order imbalance, volatility, spread).

    action = w^T * features
"""

import numpy as np


class GLFTExpert:
    """Generalized Linear Feature-based Trading expert."""

    def __init__(
        self,
        n_features: int = 6,
        risk_aversion: float = 0.1,
        max_position: float = 10.0,
    ):
        self.n_features = n_features
        self.risk_aversion = risk_aversion
        self.max_position = max_position
        # Learnable weight vector (pre-trained or set heuristically)
        self.weights = np.zeros(n_features)
        # Heuristic initialisation: penalise inventory, reward mean-reversion
        if n_features >= 4:
            self.weights[0] = -0.5   # inventory
            self.weights[1] = 0.3    # mid-price change (mean-reversion)
            self.weights[2] = -0.2   # spread
            self.weights[3] = 0.1    # volatility

    def extract_features(self, state: dict) -> np.ndarray:
        """Extract feature vector from market state.

        Expected state keys: inventory, mid_price, prev_mid_price,
        spread, volatility, order_imbalance, ...
        """
        features = np.zeros(self.n_features)
        features[0] = state.get("inventory", 0.0) / max(self.max_position, 1.0)
        features[1] = state.get("mid_price_change", 0.0)
        features[2] = state.get("spread", 0.01)
        features[3] = state.get("volatility", 0.02)
        if self.n_features > 4:
            features[4] = state.get("order_imbalance", 0.0)
        if self.n_features > 5:
            features[5] = state.get("hawkes_intensity", 5.0) / 20.0
        return features

    def act(self, state: dict) -> dict:
        """Compute action from market state.

        Returns:
            dict with target_position, weights, features
        """
        features = self.extract_features(state)
        raw_action = float(np.dot(self.weights, features))
        # Clip to position limits
        target_position = np.clip(
            raw_action * self.max_position,
            -self.max_position,
            self.max_position,
        )
        return {
            "target_position": target_position,
            "raw_action": raw_action,
            "weights": self.weights.copy(),
            "features": features,
        }
