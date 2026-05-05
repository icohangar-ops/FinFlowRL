#!/usr/bin/env python3
"""CLI: Evaluate a trained FinFlowRL policy."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np
from finflowrl.envs.hft_env import HFTEnv
from finflowrl.models.meanflow import MeanFlowPolicy
from finflowrl.evaluation.metrics import compute_pnl, compute_sharpe_ratio, compute_max_drawdown


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinFlowRL policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/meanflow_params.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load policy
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint, "r") as f:
            raw = json.load(f)
        params = {}
        for k, v in raw.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                params[k] = [np.array(x) for x in v]
            else:
                params[k] = np.array(v)
        policy.set_params(params)
        print(f"[Eval] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[Eval] No checkpoint found at {args.checkpoint}, using random policy")

    env = HFTEnv(max_steps=args.steps, seed=args.seed)
    all_returns = []

    for ep in range(args.episodes):
        obs = env.reset()
        ep_returns = []
        for _ in range(args.steps):
            action = policy.act(obs)
            obs, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
            ep_returns.append(reward)
            if done:
                break
        all_returns.extend(ep_returns)

    pnl = compute_pnl(all_returns)
    sharpe = compute_sharpe_ratio(all_returns)
    mdd = compute_max_drawdown(all_returns)

    print(f"\n{'='*40}")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Total steps:    {len(all_returns)}")
    print(f"  Cumulative PnL: {pnl:.4f}")
    print(f"  Sharpe Ratio:   {sharpe:.4f}")
    print(f"  Max Drawdown:   {mdd:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
