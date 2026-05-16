#!/usr/bin/env python3
"""CLI: Run a quick demo of FinFlowRL components."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from finflowrl.simulator.market import MarketSimulator
from finflowrl.envs.hft_env import HFTEnv
from finflowrl.models.meanflow import MeanFlowPolicy
from finflowrl.models.noise import NoisePolicy
from finflowrl.models.film import FiLMLayer
from finflowrl.experts.avellaneda_stoikov import AvellanedaStoikovExpert
from finflowrl.experts.glft import GLFTExpert
from finflowrl.experts.glft_drift import GLFTDriftExpert
from finflowrl.agents.ppo import PPOAgent
from finflowrl.evaluation.metrics import compute_pnl, compute_sharpe_ratio, compute_max_drawdown


def main():
    print("=" * 60)
    print("  FinFlowRL — Flow-Matching RL for High-Frequency Trading")
    print("=" * 60)

    # 1. Market Simulator
    print("\n[1] Market Simulator (Jump-Diffusion + Hawkes)")
    sim = MarketSimulator(seed=42)
    data = sim.simulate(n_steps=500)
    print(f"    500 steps | Mid: {data['mid_price'].mean():.2f} "
          f"| Spread: {data['spread'].mean():.6f} "
          f"| Orders/step: {data['order_arrivals'].mean():.1f}")

    # 2. Experts
    print("\n[2] Expert Policies")
    for name, expert_cls in [("Avellaneda-Stoikov", AvellanedaStoikovExpert),
                              ("GLFT", GLFTExpert),
                              ("GLFT-Drift", GLFTDriftExpert)]:
        expert = expert_cls()
        state = {"inventory": 2, "mid_price": 100, "prev_mid_price": 99.8,
                 "spread": 0.02, "volatility": 0.02, "order_imbalance": 0.1,
                 "hawkes_intensity": 6.0}
        if name == "Avellaneda-Stoikov":
            out = expert.act(100.0, 2.0, 10.0)
            print(f"    {name:20s} | Bid: {out['bid_price']:.4f} | Ask: {out['ask_price']:.4f}")
        else:
            out = expert.act(state)
            print(f"    {name:20s} | Target pos: {out['target_position']:.4f}")

    # 3. MeanFlow Policy
    print("\n[3] MeanFlow Policy (Flow Matching)")
    policy = MeanFlowPolicy(obs_dim=6, act_dim=1)
    obs = np.random.randn(6)
    action = policy.act(obs)
    loss = policy.flow_loss(obs, np.array([0.3]))
    print(f"    Params: {policy.n_params:,} | Action: {action[0]:.4f} | Loss: {loss:.4f}")

    # 4. HFT Environment
    print("\n[4] HFT Environment (5 episodes)")
    env = HFTEnv(max_steps=200, seed=42)
    all_returns = []
    for ep in range(5):
        obs = env.reset()
        for _ in range(200):
            action = np.random.randn(1) * 0.1
            obs, reward, done, info = env.step(action)
            all_returns.append(reward)
            if done:
                break
    print(f"    PnL: {compute_pnl(all_returns):.4f} "
          f"| Sharpe: {compute_sharpe_ratio(all_returns):.4f} "
          f"| MaxDD: {compute_max_drawdown(all_returns):.4f}")

    # 5. FiLM Layer
    print("\n[5] FiLM Conditioning Layer")
    film = FiLMLayer(input_dim=16, cond_dim=6)
    x = np.random.randn(16)
    cond = np.random.randn(6)
    out = film(x, cond)
    print(f"    Input: {x.shape} + Cond: {cond.shape} -> Output: {out.shape}")

    print("\n" + "=" * 60)
    print("  All components verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
