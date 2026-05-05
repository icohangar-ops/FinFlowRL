"""Configuration system for FinFlowRL experiments."""

import yaml
import os
from typing import Optional, Dict, Any


_DEFAULT_CONFIG = {
    "simulator": {
        "seed": 42,
        "S0": 100.0,
        "mu": 0.0,
        "sigma": 0.02,
        "jump_intensity": 0.1,
        "jump_mean": 0.0,
        "jump_std": 0.01,
        "half_spread": 0.01,
        "hawkes_mu": 5.0,
        "hawkes_alpha": 2.0,
        "hawkes_beta": 10.0,
        "dt": 1.0,
    },
    "env": {
        "max_steps": 1000,
        "max_position": 10,
        "transaction_cost": 0.001,
        "inventory_penalty": 0.01,
        "reward_scale": 1.0,
        "obs_dim": 6,
        "act_dim": 1,
    },
    "policy": {
        "obs_dim": 6,
        "act_dim": 1,
        "hidden_sizes": [128, 128, 64],
        "n_flow_steps": 10,
    },
    "pretrain": {
        "n_episodes": 100,
        "steps_per_episode": 200,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "n_iterations": 1000,
    },
    "finetune": {
        "n_episodes": 50,
        "steps_per_episode": 500,
        "n_epochs": 10,
    },
    "ppo": {
        "hidden_sizes": [64, 64],
        "lr": 3e-4,
        "clip_ratio": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
    },
    "expert": {
        "type": "glft",
        "gamma": 0.1,
        "sigma": 0.02,
    },
}


class Config:
    """YAML-based configuration for FinFlowRL experiments."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = _DEFAULT_CONFIG.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-separated key (e.g. 'simulator.seed')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set config value by dot-separated key."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    @property
    def raw(self) -> Dict:
        """Return raw config dict."""
        return self._config

    def __repr__(self) -> str:
        return f"Config({self._config})"
