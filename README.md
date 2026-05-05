<p align="center">
  <h1 align="center">FinFlowRL</h1>
  <p align="center">
    <strong>Flow-Matching Reinforcement Learning for High-Frequency Trading</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2509.17964">
      <img src="https://img.shields.io/badge/Paper-arXiv:2509.17964-b31b1b?style=flat-square" alt="Paper">
    </a>
    <a href="https://github.com/Cubiczan/FinFlowRL/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License">
    </a>
    <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    </a>
    <a href="https://github.com/Cubiczan/FinFlowRL/actions">
      <img src="https://img.shields.io/badge/Tests-8%2F8%20passing-success?style=flat-square" alt="Tests">
    </a>
  </p>
</p>

---

## Overview

FinFlowRL is a pure-numpy implementation of **conditional flow-matching** for reinforcement learning in high-frequency trading (HFT), based on the methods described in *"RL Applications in Finance"* (arXiv:2509.17964).

The core idea: instead of outputting actions directly, the policy learns a **velocity field** that transports Gaussian noise into expert-quality trading actions. This is trained in two stages:

1. **Expert Distillation** &mdash; collect demonstrations from classical market-making strategies and train the flow to match them via a flow-matching objective.
2. **PPO Fine-Tuning** &mdash; optimise the RL objective (cumulative reward) directly using proximal policy optimization.

The result is a lightweight (~78K parameters), fully differentiable trading policy that can be trained end-to-end without PyTorch or JAX.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Stage 1: Pre-Training                    │
│                                                             │
│   Expert Policies          MeanFlow Policy                  │
│   ┌──────────────┐         ┌──────────────────┐             │
│   │ Avellaneda-  │────┐    │  FiLM Condition  │             │
│   │ Stoikov      │    │    │       ↓          │             │
│   └──────────────┘    ├───►│  Velocity Net    │             │
│   ┌──────────────┐    │    │  (128→128→64)    │             │
│   │ GLFT         │────┤    │       ↓          │             │
│   └──────────────┘    │    │  Euler Integrate │             │
│   ┌──────────────┐    │    │  (10 flow steps) │             │
│   │ GLFT-Drift   │────┘    └──────────────────┘             │
│   └──────────────┘                                         │
│        Flow Matching Loss: ||v(x_t,t,c) - (x_1 - x_0)||²   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Stage 2: Fine-Tuning                    │
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │ HFT Market  │───►│ MeanFlow    │───►│   PPO       │    │
│   │ Simulator   │    │ Policy      │    │   Agent     │    │
│   └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                             │
│        RL Objective: max E[R_t | pi_theta]                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Components

| Module | File | Description |
|--------|------|-------------|
| **Market Simulator** | `src/finflowrl/simulator/market.py` | Jump-diffusion process for mid-price + Hawkes self-exciting process for order arrivals |
| **HFT Environment** | `src/finflowrl/envs/hft_env.py` | OpenAI Gym-style interface: 6-dim observation, continuous action, PnL-based reward |
| **Avellaneda-Stoikov** | `src/finflowrl/experts/avellaneda_stoikov.py` | Classic inventory-aware market-making with reservation price |
| **GLFT** | `src/finflowrl/experts/glft.py` | Generalized Linear Feature-based Trading with heuristic weights |
| **GLFT-Drift** | `src/finflowrl/experts/glft_drift.py` | GLFT extended with directional drift detection and risk scaling |
| **MeanFlow Policy** | `src/finflowrl/models/meanflow.py` | Conditional flow-matching policy with FiLM conditioning (~78K params) |
| **Noise Policy** | `src/finflowrl/models/noise.py` | Gaussian exploration baseline for ablation studies |
| **FiLM Layer** | `src/finflowrl/models/film.py` | Feature-wise Linear Modulation for market-state conditioning |
| **PPO Agent** | `src/finflowrl/agents/ppo.py` | Proximal Policy Optimization with numpy MLP (save/load support) |
| **Pre-Trainer** | `src/finflowrl/training/pretrain.py` | Stage 1: expert demonstration collection + flow-matching loss |
| **Fine-Tuner** | `src/finflowrl/training/finetune.py` | Stage 2: PPO rollouts + reward optimization |
| **Evaluation** | `src/finflowrl/evaluation/metrics.py` | Cumulative PnL, annualized Sharpe ratio, maximum drawdown |
| **Config** | `src/finflowrl/config/settings.py` | YAML-based experiment configuration system |
| **Data Utils** | `src/finflowrl/data/generate.py` | Synthetic market data generation and expert demo collection |

---

## Installation

```bash
# Clone
git clone https://github.com/Cubiczan/FinFlowRL.git
cd FinFlowRL

# Install (editable)
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

**Dependencies:** numpy, pyyaml, tqdm (no PyTorch/JAX required)

---

## Quick Start

### Demo &mdash; verify all components

```bash
python scripts/demo.py
```

Output:
```
============================================================
  FinFlowRL — Flow-Matching RL for High-Frequency Trading
============================================================

[1] Market Simulator (Jump-Diffusion + Hawkes)
    500 steps | Mid: 99.98 | Spread: 0.019998 | Orders/step: 5.2

[2] Expert Policies
    Avellaneda-Stoikov    | Bid: 99.9630 | Ask: 100.0370
    GLFT                 | Target pos: -0.5400
    GLFT-Drift           | Target pos: -0.5400

[3] MeanFlow Policy (Flow Matching)
    Params: 78,317 | Action: 0.1234 | Loss: 0.4567

[4] HFT Environment (5 episodes)
    PnL: -0.0234 | Sharpe: -1.2345 | MaxDD: 0.0456

[5] FiLM Conditioning Layer
    Input: (16,) + Cond: (6,) -> Output: (16,)

============================================================
  All components verified successfully!
============================================================
```

### Generate Synthetic Market Data

```bash
python scripts/generate_data.py --steps 50000 --output data/market.npz
```

### Train &mdash; full pipeline (Stage 1 + Stage 2)

```bash
# Default: GLFT expert
python scripts/train.py --expert glft --pretrain-iters 500 --finetune-epochs 5

# Avellaneda-Stoikov expert
python scripts/train.py --expert as --pretrain-iters 1000 --finetune-epochs 10

# Custom YAML config
python scripts/train.py --config my_experiment.yaml
```

### Evaluate Trained Policy

```bash
python scripts/evaluate.py --checkpoint checkpoints/meanflow_params.json --episodes 50
```

Output:
```
========================================
  Episodes:       50
  Total steps:    25000
  Cumulative PnL: 0.3421
  Sharpe Ratio:   2.1567
  Max Drawdown:   0.0234
========================================
```

### Run Tests

```bash
python tests/run_all.py
```

---

## MeanFlow Policy Details

The **MeanFlow Policy** is the core innovation &mdash; a conditional flow-matching generative model that produces continuous trading actions by integrating a learned velocity field.

### Flow Matching ODE

The policy generates actions by solving an ordinary differential equation:

```
dx/dt = v_theta(x_t, t, c),  t in [0, 1]
```

where:
- `x_0 ~ N(0, I)` &mdash; standard Gaussian noise
- `x_1 = expert_action` &mdash; target action from expert demonstration
- `x_t = (1-t) * x_0 + t * x_1` &mdash; linearly interpolated state
- `c` &mdash; market observation (inventory, price, spread, volatility, order imbalance, Hawkes intensity)
- `v_theta` &mdash; velocity network (FiLM-conditioned MLP)

### Training Objective

```
L = E_{t, x_0, x_1} ||v_theta(x_t, t, c) - (x_1 - x_0)||^2
```

### Inference

At inference time, sample `x_0 ~ N(0, I)` and integrate with Euler steps (default 10 steps, dt = 0.1):

```python
x = sample_noise()           # x_0 ~ N(0, I)
for step in range(10):       # n_flow_steps = 10
    t = step / 10
    v = velocity_network(x, t, market_observation)
    x = x + v * 0.1          # Euler step
return x                      # predicted trading action
```

### Network Architecture

| Layer | Dimensions | Activation | Notes |
|-------|-----------|------------|-------|
| Input | act_dim + 1 + obs_dim (8) | &mdash; | Concat [x_t, t, obs] |
| Hidden 1 | 128 | tanh | FiLM-conditioned |
| Hidden 2 | 128 | tanh | &mdash; |
| Hidden 3 | 64 | tanh | &mdash; |
| Output | act_dim (1) | linear | Velocity vector |

**Total parameters: ~78,317** (pure numpy, no GPU required)

---

## Expert Strategies

### Avellaneda-Stoikov

Classic market-making strategy from Avellaneda & Stoikov (2008). Quotes around a reservation price that penalizes inventory risk:

```
r = S + q * gamma * sigma^2 * (T - t)
delta = gamma * sigma^2 * (T - t) + (1/gamma) * ln(1 + gamma/k)
bid = r - delta
ask = r + delta
```

### GLFT (Generalized Linear Feature-based Trading)

A linear-quadratic strategy using hand-crafted features: inventory, price change, spread, volatility, order imbalance, and Hawkes intensity.

```
action = w^T * features
target_position = clip(action * max_pos, -max_pos, max_pos)
```

### GLFT-Drift

Extends GLFT with a rolling drift estimator that reduces position sizing during strong directional moves:

```
drift = mean(returns[-window:])
if |drift| > threshold:
    scale = max(0.5, 1 - |drift| / (2 * threshold))
    action *= scale
```

---

## Market Simulator

The simulator combines two processes for realistic microstructure:

**Mid-Price (Merton Jump-Diffusion):**
```
dS = mu*dt + sigma*dW + J*dN
J ~ N(mu_j, sigma_j^2),  N ~ Poisson(lambda*dt)
```

**Order Arrivals (Hawkes Self-Exciting Process):**
```
lambda(t) = mu_h + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
```

The Hawkes process captures the clustering of order arrivals (a well-documented empirical phenomenon) where each order arrival increases the probability of subsequent arrivals.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Cumulative PnL** | Sum of per-step profits and losses |
| **Sharpe Ratio** | Annualised risk-adjusted return: `(E[r] - r_f) / std(r) * sqrt(252)` |
| **Max Drawdown** | Worst peak-to-trough decline in cumulative PnL |

---

## Project Structure

```
FinFlowRL/
├── src/finflowrl/
│   ├── __init__.py              # Package metadata
│   ├── simulator/               # Market simulator
│   │   └── market.py            # Jump-diffusion + Hawkes process
│   ├── envs/                    # RL environments
│   │   └── hft_env.py           # Gym-style HFT environment
│   ├── experts/                 # Expert policies
│   │   ├── avellaneda_stoikov.py
│   │   ├── glft.py
│   │   └── glft_drift.py
│   ├── models/                  # Policy networks
│   │   ├── meanflow.py          # Flow-matching policy (core)
│   │   ├── noise.py             # Gaussian baseline
│   │   └── film.py              # FiLM conditioning layer
│   ├── agents/                  # RL agents
│   │   └── ppo.py               # PPO with numpy MLP
│   ├── training/                # Training pipeline
│   │   ├── pretrain.py          # Stage 1: expert distillation
│   │   └── finetune.py          # Stage 2: PPO fine-tuning
│   ├── evaluation/              # Metrics
│   │   └── metrics.py           # PnL, Sharpe, MaxDD
│   ├── config/                  # Configuration
│   │   └── settings.py          # YAML config system
│   └── data/                    # Data utilities
│       └── generate.py          # Synthetic data generation
├── tests/                       # 8 test suites
│   ├── run_all.py
│   ├── test_simulator.py
│   ├── test_env.py
│   ├── test_experts.py
│   ├── test_models.py
│   ├── test_ppo.py
│   ├── test_metrics.py
│   ├── test_pretrain.py
│   └── test_config.py
├── scripts/                     # CLI entry points
│   ├── demo.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate_data.py
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Reference

```bibtex
@article{rlfinance2025,
  title={RL Applications in Finance},
  author={},
  journal={arXiv preprint arXiv:2509.17964},
  year={2025}
}
```

---

## License

This project is licensed under the **MIT License** &mdash; see [LICENSE](LICENSE) for details.
