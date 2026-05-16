"""Gaussian Noise Policy — simple exploration baseline.

Samples actions from a Gaussian distribution parameterised by a learned
mean, with fixed or learned standard deviation.
"""

import numpy as np


class NoisePolicy:
    """Gaussian noise exploration policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        noise_std: float = 0.1,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.noise_std = noise_std

        # Mean network weights
        scale = np.sqrt(2.0 / obs_dim)
        self.W1 = np.random.randn(obs_dim, hidden_size) * scale
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, act_dim) * scale
        self.b2 = np.zeros(act_dim)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass to get action mean."""
        x = obs.astype(np.float64) @ self.W1 + self.b1
        x = np.tanh(x)
        x = x @ self.W2 + self.b2
        return x

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Sample action: mean + Gaussian noise."""
        mean = self.forward(obs)
        noise = np.random.randn(self.act_dim) * self.noise_std
        return mean + noise

    def get_mean(self, obs: np.ndarray) -> np.ndarray:
        """Get deterministic action (mean without noise)."""
        return self.forward(obs)
