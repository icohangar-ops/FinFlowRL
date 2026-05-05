"""Tests for Config system."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile
from finflowrl.config.settings import Config


def test_default_config():
    cfg = Config()
    assert cfg.get("simulator.seed") == 42
    assert cfg.get("env.max_steps") == 1000
    assert cfg.get("policy.obs_dim") == 6


def test_config_set():
    cfg = Config()
    cfg.set("simulator.sigma", 0.05)
    assert cfg.get("simulator.sigma") == 0.05


def test_config_missing_key():
    cfg = Config()
    assert cfg.get("nonexistent.key") is None
    assert cfg.get("nonexistent.key", 42) == 42


def test_config_save_load():
    cfg = Config()
    cfg.set("simulator.sigma", 0.99)
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        path = f.name
    cfg.save(path)
    cfg2 = Config(path)
    assert cfg2.get("simulator.sigma") == 0.99
    os.unlink(path)


if __name__ == "__main__":
    test_default_config()
    test_config_set()
    test_config_missing_key()
    test_config_save_load()
    print("All config tests passed!")
