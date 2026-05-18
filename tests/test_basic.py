"""Basic import tests for FinFlowRL.

Validates that core modules can be imported without errors.
"""

def test_import_package():
    """Test that the FinFlowRL package imports with version."""
    from finflowrl import __version__, __author__
    assert __version__ == "0.1.0"
    assert "FinFlowRL" in __author__


def test_import_config():
    """Test that the config module imports."""
    from finflowrl.config.settings import Config
    assert Config is not None


def test_config_defaults():
    """Test that Config loads defaults correctly."""
    from finflowrl.config.settings import Config
    cfg = Config()
    assert cfg.get("simulator.seed") == 42
    assert cfg.get("simulator.sigma") == 0.02
    assert cfg.get("env.max_steps") == 1000


def test_config_get_missing_key():
    """Test that Config.get returns default for missing keys."""
    from finflowrl.config.settings import Config
    cfg = Config()
    assert cfg.get("nonexistent.key", "fallback") == "fallback"


def test_config_set_and_get():
    """Test that Config.set and Config.get work together."""
    from finflowrl.config.settings import Config
    cfg = Config()
    cfg.set("custom.value", 99)
    assert cfg.get("custom.value") == 99


def test_import_simulator():
    """Test that the simulator module imports."""
    import finflowrl.simulator
    assert finflowrl.simulator is not None


def test_import_models():
    """Test that the models module imports."""
    import finflowrl.models
    assert finflowrl.models is not None


def test_import_evaluation():
    """Test that the evaluation module imports."""
    import finflowrl.evaluation
    assert finflowrl.evaluation is not None


def test_import_envs():
    """Test that the envs module imports."""
    import finflowrl.envs
    assert finflowrl.envs is not None


def test_import_experts():
    """Test that the experts module imports."""
    import finflowrl.experts
    assert finflowrl.experts is not None


def test_import_agents():
    """Test that the agents module imports."""
    import finflowrl.agents
    assert finflowrl.agents is not None
