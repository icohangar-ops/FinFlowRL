"""Tests for MarketSimulator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from finflowrl.simulator.market import MarketSimulator


def test_simulator_creation():
    sim = MarketSimulator(seed=42)
    assert sim.S0 == 100.0
    assert sim.sigma == 0.02


def test_simulator_reset():
    sim = MarketSimulator(seed=42)
    price = sim.reset(S0=150.0)
    assert price == 150.0
    assert sim.mid_price == 150.0


def test_simulator_step():
    sim = MarketSimulator(seed=42)
    sim.reset()
    state = sim.step()
    assert "mid_price" in state
    assert "best_bid" in state
    assert "best_ask" in state
    assert state["best_bid"] < state["best_ask"]
    assert state["mid_price"] > 0


def test_simulator_simulate():
    sim = MarketSimulator(seed=42)
    data = sim.simulate(n_steps=100)
    assert len(data["mid_price"]) == 100
    assert len(data["best_bid"]) == 100
    assert np.all(data["mid_price"] > 0)
    assert np.all(data["spread"] > 0)


def test_simulator_deterministic():
    sim1 = MarketSimulator(seed=123)
    sim2 = MarketSimulator(seed=123)
    d1 = sim1.simulate(50)
    d2 = sim2.simulate(50)
    np.testing.assert_array_almost_equal(d1["mid_price"], d2["mid_price"])


if __name__ == "__main__":
    test_simulator_creation()
    test_simulator_reset()
    test_simulator_step()
    test_simulator_simulate()
    test_simulator_deterministic()
    print("All simulator tests passed!")
