#!/usr/bin/env python3
"""CLI: Train FinFlowRL (pre-train + fine-tune)."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finflowrl.config.settings import Config
from finflowrl.envs.hft_env import HFTEnv
from finflowrl.models.meanflow import MeanFlowPolicy
from finflowrl.agents.ppo import PPOAgent
from finflowrl.experts.glft import GLFTExpert
from finflowrl.experts.avellaneda_stoikov import AvellanedaStoikovExpert
from finflowrl.experts.glft_drift import GLFTDriftExpert
from finflowrl.training.pretrain import PreTrainer
from finflowrl.training.finetune import FineTuner


EXPERTS = {"as": AvellanedaStoikovExpert, "glft": GLFTExpert, "glft_drift": GLFTDriftExpert}


def main():
    parser = argparse.ArgumentParser(description="Train FinFlowRL")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--expert", type=str, default="glft", choices=list(EXPERTS.keys()))
    parser.add_argument("--pretrain-iters", type=int, default=500)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--output", type=str, default="checkpoints", help="Output dir")
    args = parser.parse_args()

    cfg = Config(args.config)
    print(f"[FinFlowRL] Config: expert={args.expert}, pretrain={args.pretrain_iters}, finetune={args.finetune_epochs}")

    os.makedirs(args.output, exist_ok=True)

    # Build components
    env = HFTEnv(
        max_steps=cfg.get("pretrain.steps_per_episode", 200),
        max_position=cfg.get("env.max_position", 10),
        seed=cfg.get("simulator.seed", 42),
    )
    policy = MeanFlowPolicy(
        obs_dim=cfg.get("policy.obs_dim", 6),
        act_dim=cfg.get("policy.act_dim", 1),
        hidden_sizes=tuple(cfg.get("policy.hidden_sizes", [128, 128, 64])),
    )
    print(f"[FinFlowRL] MeanFlow params: {policy.n_params:,}")

    expert = EXPERTS[args.expert]()

    # Stage 1: Pre-train (expert distillation)
    print("\n=== Stage 1: Expert Distillation ===")
    pretrainer = PreTrainer(
        policy, env, expert,
        n_episodes=cfg.get("pretrain.n_episodes", 50),
        steps_per_episode=cfg.get("pretrain.steps_per_episode", 200),
    )
    pretrainer.train(n_iterations=args.pretrain_iters)

    # Stage 2: Fine-tune (PPO)
    print("\n=== Stage 2: PPO Fine-tuning ===")
    ppo = PPOAgent(obs_dim=cfg.get("policy.obs_dim", 6), act_dim=3, hidden_sizes=(64, 64))
    finetuner = FineTuner(policy, env, ppo, n_episodes=10, steps_per_episode=200)
    finetuner.train(n_epochs=args.finetune_epochs)

    # Save policy
    import json
    params = policy.get_params()
    saveable = {}
    for k, v in params.items():
        if isinstance(v, list):
            saveable[k] = [x.tolist() for x in v]
        elif isinstance(v, np.ndarray):
            saveable[k] = v.tolist()
    with open(os.path.join(args.output, "meanflow_params.json"), "w") as f:
        json.dump(saveable, f)
    ppo.save(os.path.join(args.output, "ppo_weights.json"))

    print(f"\n[FinFlowRL] Done. Checkpoints saved to {args.output}/")


if __name__ == "__main__":
    main()
