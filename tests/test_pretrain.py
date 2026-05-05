"""Tests for Pre-Trainer and Fine-Tuner."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from finflowrl.envs.hft_env import HFTEnv
from finflowrl.models.meanflow import MeanFlowPolicy
from finflowrl.experts.glft import GLFTExpert
from finflowrl.agents.ppo import PPOAgent
from finflowrl.training.pretrain import PreTrainer
from finflowrl.training.finetune import FineTuner


def test_pretrainer_creation():
    env = HFTEnv(seed=42, max_steps=20)
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    expert = GLFTExpert(n_features=6)
    trainer = PreTrainer(policy, env, expert, n_episodes=2, steps_per_episode=10)
    assert trainer is not None


def test_pretrainer_collect():
    env = HFTEnv(seed=42, max_steps=20)
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    expert = GLFTExpert(n_features=6)
    trainer = PreTrainer(policy, env, expert, n_episodes=2, steps_per_episode=10)
    trainer.collect_expert_demos()
    assert len(trainer.obs_buffer) > 0
    assert len(trainer.action_buffer) > 0


def test_pretrainer_train_step():
    env = HFTEnv(seed=42, max_steps=20)
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    expert = GLFTExpert(n_features=6)
    trainer = PreTrainer(policy, env, expert, n_episodes=2, steps_per_episode=10)
    trainer.collect_expert_demos()
    loss = trainer.train_step()
    assert np.isfinite(loss)


def test_finetuner_creation():
    env = HFTEnv(seed=42, max_steps=20)
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    ppo = PPOAgent(obs_dim=6, act_dim=3, hidden_sizes=(32, 32))
    ft = FineTuner(policy, env, ppo, n_episodes=1, steps_per_episode=10)
    assert ft is not None


def test_finetuner_rollout():
    env = HFTEnv(seed=42, max_steps=20)
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    ppo = PPOAgent(obs_dim=6, act_dim=3, hidden_sizes=(32, 32))
    ft = FineTuner(policy, env, ppo, n_episodes=1, steps_per_episode=10)
    rollout = ft.collect_rollout()
    assert "obs" in rollout
    assert "rewards" in rollout
    assert len(rollout["obs"]) > 0


if __name__ == "__main__":
    test_pretrainer_creation()
    test_pretrainer_collect()
    test_pretrainer_train_step()
    test_finetuner_creation()
    test_finetuner_rollout()
    print("All training tests passed!")
