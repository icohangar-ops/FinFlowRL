"""PPO Agent — Proximal Policy Optimization with a numpy MLP.

Lightweight PPO implementation for fine-tuning the MeanFlow policy.
Supports save/load and on-policy rollouts.
"""

import numpy as np
import json
from typing import Optional, Tuple, List


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class MLPPolicy:
    """Simple multi-layer perceptron policy network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...] = (64, 64)):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1  # hidden layers + output

        # Xavier initialization
        self.weights: list = []
        self.biases: list = []
        prev_dim = obs_dim
        for h in hidden_sizes:
            w = np.random.randn(prev_dim, h) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(h)
            self.weights.append(w)
            self.biases.append(b)
            prev_dim = h
        # Output layer (logits for discrete, values for continuous)
        w_out = np.random.randn(prev_dim, act_dim) * np.sqrt(2.0 / prev_dim)
        b_out = np.zeros(act_dim)
        self.weights.append(w_out)
        self.biases.append(b_out)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass. Returns action logits/values."""
        x = obs.astype(np.float64)
        for i in range(self.n_layers):
            x = x @ self.weights[i] + self.biases[i]
            if i < self.n_layers - 1:
                x = np.tanh(x)
        return x

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Sample action from policy.

        Returns:
            (action, log_prob)
        """
        logits = self.forward(obs)
        probs = _softmax(logits)
        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(len(probs), p=probs)
        log_prob = float(np.log(probs[action] + 1e-10))
        return action, log_prob

    def get_value(self, obs: np.ndarray) -> float:
        """Get scalar value estimate (uses first output neuron)."""
        logits = self.forward(obs)
        return float(logits[0])


class PPOAgent:
    """PPO agent wrapping an MLP policy with clip-based updates."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.policy = MLPPolicy(obs_dim, act_dim, hidden_sizes)
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch_size = minibatch_size

    def save(self, path: str) -> None:
        """Save policy weights to JSON file."""
        data = {
            "obs_dim": self.policy.obs_dim,
            "act_dim": self.policy.act_dim,
            "hidden_sizes": list(self.policy.hidden_sizes),
            "weights": [w.tolist() for w in self.policy.weights],
            "biases": [b.tolist() for b in self.policy.biases],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load policy weights from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        assert data["obs_dim"] == self.policy.obs_dim
        assert data["act_dim"] == self.policy.act_dim
        for i, w in enumerate(data["weights"]):
            self.policy.weights[i] = np.array(w)
        for i, b in enumerate(data["biases"]):
            self.policy.biases[i] = np.array(b)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Select action given observation.

        Returns:
            (action, log_prob)
        """
        return self.policy.get_action(obs, deterministic)

    @staticmethod
    def compute_gae(rewards: list, values: list, dones: list) -> Tuple[list, list]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1]
            delta = rewards[t] + 0.99 * next_val * (1 - dones[t]) - values[t]
            gae = delta + 0.99 * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns
