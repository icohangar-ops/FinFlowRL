"""GLFT-Drift expert — extends GLFT with a learned drift correction term.

Adds a drift-aware component that adapts the quoting strategy when the
mid-price exhibits directional momentum.
"""

import numpy as np


class GLFTDriftExpert:
    """GLFT with drift correction for trending markets."""

    def __init__(
        self,
        n_features: int = 8,
        risk_aversion: float = 0.1,
        max_position: float = 10.0,
        drift_window: int = 20,
        drift_threshold: float = 0.005,
    ):
        self.n_features = n_features
        self.risk_aversion = risk_aversion
        self.max_position = max_position
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold

        self.weights = np.zeros(n_features)
        # GLFT base weights
        if n_features >= 6:
            self.weights[0] = -0.5
            self.weights[1] = 0.3
            self.weights[2] = -0.2
            self.weights[3] = 0.1
            self.weights[4] = -0.15   # drift correction
            self.weights[5] = -0.1    # drift squared (nonlinear)

        self.price_history: list = []

    def compute_drift(self) -> float:
        """Compute rolling drift from price history."""
        if len(self.price_history) < 2:
            return 0.0
        window = self.price_history[-self.drift_window:]
        if len(window) < 2:
            return 0.0
        returns = np.diff(window) / window[:-1]
        return float(np.mean(returns))

    def extract_features(self, state: dict) -> np.ndarray:
        """Extract extended feature vector including drift terms."""
        features = np.zeros(self.n_features)
        features[0] = state.get("inventory", 0.0) / max(self.max_position, 1.0)
        features[1] = state.get("mid_price_change", 0.0)
        features[2] = state.get("spread", 0.01)
        features[3] = state.get("volatility", 0.02)
        if self.n_features > 4:
            features[4] = state.get("order_imbalance", 0.0)
        if self.n_features > 5:
            features[5] = state.get("hawkes_intensity", 5.0) / 20.0
        # Drift features
        drift = self.compute_drift()
        if self.n_features > 6:
            features[6] = drift
        if self.n_features > 7:
            features[7] = drift ** 2
        return features

    def act(self, state: dict) -> dict:
        """Compute drift-aware action.

        Returns:
            dict with target_position, drift, weights, features
        """
        mid_price = state.get("mid_price", 100.0)
        self.price_history.append(mid_price)
        # Keep bounded history
        if len(self.price_history) > 2000:
            self.price_history = self.price_history[-2000:]

        features = self.extract_features(state)
        drift = self.compute_drift()

        raw_action = float(np.dot(self.weights, features))

        # Reduce position in strong drift (risk management)
        if abs(drift) > self.drift_threshold:
            scale = max(0.5, 1.0 - abs(drift) / (2.0 * self.drift_threshold))
            raw_action *= scale

        target_position = np.clip(
            raw_action * self.max_position,
            -self.max_position,
            self.max_position,
        )
        return {
            "target_position": target_position,
            "raw_action": raw_action,
            "drift": drift,
            "weights": self.weights.copy(),
            "features": features,
        }
