"""Tests for Expert Policies."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from finflowrl.experts.avellaneda_stoikov import AvellanedaStoikovExpert
from finflowrl.experts.glft import GLFTExpert
from finflowrl.experts.glft_drift import GLFTDriftExpert


def test_as_expert():
    expert = AvellanedaStoikovExpert(gamma=0.1, sigma=0.02)
    result = expert.act(mid_price=100.0, inventory=0.0, t=10.0)
    assert "bid_price" in result
    assert "ask_price" in result
    assert result["bid_price"] < result["ask_price"]


def test_as_reservation_price():
    expert = AvellanedaStoikovExpert(gamma=0.1, sigma=0.02)
    r = expert.get_reservation_price(100.0, 5.0, 10.0)
    assert r < 100.0  # long inventory -> lower reservation


def test_glft_expert():
    expert = GLFTExpert(n_features=6)
    state = {
        "inventory": 3.0, "mid_price": 100.0, "prev_mid_price": 99.5,
        "spread": 0.02, "volatility": 0.02, "order_imbalance": 0.1,
    }
    result = expert.act(state)
    assert "target_position" in result
    assert abs(result["target_position"]) <= 10.0


def test_glft_features():
    expert = GLFTExpert(n_features=6)
    state = {"inventory": 0, "mid_price_change": 0, "spread": 0.01,
             "volatility": 0.02, "order_imbalance": 0, "hawkes_intensity": 5}
    features = expert.extract_features(state)
    assert features.shape == (6,)


def test_glft_drift_expert():
    expert = GLFTDriftExpert(n_features=8)
    state = {
        "inventory": 2.0, "mid_price": 100.0, "mid_price_change": 0.001,
        "spread": 0.02, "volatility": 0.02, "order_imbalance": 0.1,
        "hawkes_intensity": 5.0,
    }
    result = expert.act(state)
    assert "drift" in result
    assert "target_position" in result
    assert abs(result["target_position"]) <= 10.0


if __name__ == "__main__":
    test_as_expert()
    test_as_reservation_price()
    test_glft_expert()
    test_glft_features()
    test_glft_drift_expert()
    print("All expert tests passed!")
