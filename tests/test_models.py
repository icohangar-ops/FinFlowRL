"""Tests for MeanFlow Policy, NoisePolicy, and FiLM."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from finflowrl.models.meanflow import MeanFlowPolicy
from finflowrl.models.noise import NoisePolicy
from finflowrl.models.film import FiLMLayer


def test_meanflow_creation():
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    assert policy.obs_dim == 6
    assert policy.act_dim == 1
    assert policy.n_params > 0


def test_meanflow_params():
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    params = policy.get_params()
    assert "vel_weights" in params
    policy2 = MeanFlowPolicy(obs_dim=6, act_dim=1)
    policy2.set_params(params)
    obs = np.random.randn(6)
    a1 = policy.act(obs, deterministic=True)
    a2 = policy2.act(obs, deterministic=True)
    np.testing.assert_array_almost_equal(a1, a2)


def test_meanflow_loss():
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    obs = np.random.randn(6)
    action = np.random.randn(1)
    loss = policy.flow_loss(obs, action)
    assert loss >= 0
    assert np.isfinite(loss)


def test_meanflow_act():
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    obs = np.random.randn(6)
    action = policy.act(obs, deterministic=False)
    assert action.shape == (1,)
    assert np.isfinite(action).all()


def test_noise_policy():
    policy = NoisePolicy(obs_dim=6, act_dim=2)
    obs = np.random.randn(6)
    action = policy.act(obs)
    assert action.shape == (2,)
    mean = policy.get_mean(obs)
    assert mean.shape == (2,)


def test_film_layer():
    film = FiLMLayer(input_dim=8, cond_dim=4)
    x = np.random.randn(8)
    cond = np.random.randn(4)
    out = film(x, cond)
    assert out.shape == (8,)
    assert np.isfinite(out).all()
    params = film.get_params()
    film2 = FiLMLayer(input_dim=8, cond_dim=4)
    film2.set_params(params)
    out2 = film2(x, cond)
    np.testing.assert_array_almost_equal(out, out2)


if __name__ == "__main__":
    test_meanflow_creation()
    test_meanflow_params()
    test_meanflow_loss()
    test_meanflow_act()
    test_noise_policy()
    test_film_layer()
    print("All model tests passed!")
