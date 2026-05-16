"""Tests for HFT Environment."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from finflowrl.envs.hft_env import HFTEnv


def test_env_creation():
    env = HFTEnv(seed=42)
    assert env.obs_dim == 6
    assert env.act_dim == 1


def test_env_reset():
    env = HFTEnv(seed=42)
    obs = env.reset()
    assert obs.shape == (6,)
    assert np.all(np.isfinite(obs))


def test_env_step():
    env = HFTEnv(seed=42)
    obs = env.reset()
    action = np.array([0.5])
    obs2, reward, done, info = env.step(action)
    assert obs2.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "pnl" in info
    assert "inventory" in info


def test_env_full_episode():
    env = HFTEnv(seed=42, max_steps=50)
    obs = env.reset()
    total_reward = 0.0
    for _ in range(50):
        action = np.random.randn(1) * 0.1
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    assert done is True


def test_env_position_clipping():
    env = HFTEnv(seed=42, max_position=5)
    obs = env.reset()
    obs, _, _, info = env.step(np.array([100.0]))
    assert abs(info["inventory"]) <= 5.0


if __name__ == "__main__":
    test_env_creation()
    test_env_reset()
    test_env_step()
    test_env_full_episode()
    test_env_position_clipping()
    print("All env tests passed!")
