"""Tests for PPO Agent."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import tempfile
from finflowrl.agents.ppo import PPOAgent, MLPPolicy


def test_mlp_creation():
    policy = MLPPolicy(obs_dim=6, act_dim=3, hidden_sizes=(32, 32))
    assert policy.obs_dim == 6
    assert policy.act_dim == 3


def test_mlp_forward():
    policy = MLPPolicy(obs_dim=6, act_dim=3, hidden_sizes=(32, 32))
    obs = np.random.randn(6)
    logits = policy.forward(obs)
    assert logits.shape == (3,)


def test_mlp_get_action():
    policy = MLPPolicy(obs_dim=6, act_dim=3, hidden_sizes=(32, 32))
    obs = np.random.randn(6)
    action, log_prob = policy.get_action(obs)
    assert 0 <= action < 3
    assert np.isfinite(log_prob)


def test_ppo_creation():
    agent = PPOAgent(obs_dim=6, act_dim=3, hidden_sizes=(32, 32))
    assert agent is not None


def test_ppo_select_action():
    agent = PPOAgent(obs_dim=6, act_dim=3, hidden_sizes=(32, 32))
    obs = np.random.randn(6)
    action, log_prob = agent.select_action(obs)
    assert 0 <= action < 3


def test_ppo_save_load():
    agent = PPOAgent(obs_dim=6, act_dim=3, hidden_sizes=(32, 32))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    agent.save(path)
    agent2 = PPOAgent(obs_dim=6, act_dim=3, hidden_sizes=(32, 32))
    agent2.load(path)
    obs = np.random.randn(6)
    a1, _ = agent.select_action(obs, deterministic=True)
    a2, _ = agent2.select_action(obs, deterministic=True)
    assert a1 == a2
    os.unlink(path)


def test_ppo_gae():
    rewards = [1.0, -0.5, 0.3, 0.8]
    values = [0.5, 0.2, 0.1, 0.0]
    dones = [0, 0, 0, 1]
    advs, rets = PPOAgent.compute_gae(rewards, values, dones)
    assert len(advs) == 4
    assert len(rets) == 4


if __name__ == "__main__":
    test_mlp_creation()
    test_mlp_forward()
    test_mlp_get_action()
    test_ppo_creation()
    test_ppo_select_action()
    test_ppo_save_load()
    test_ppo_gae()
    print("All PPO tests passed!")
