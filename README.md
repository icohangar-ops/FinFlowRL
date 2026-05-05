# FinFlowRL

**Flow-Matching Reinforcement Learning for High-Frequency Trading**

Based on *"RL Applications in Finance"* (arXiv:2509.17964), FinFlowRL implements a two-stage training pipeline that distills expert market-making strategies into a flow-matching generative policy, then fine-tunes it with PPO.

## Architecture

```
Stage 1: Expert Distillation (Flow Matching)
  Expert Policies (AS, GLFT, GLFT-Drift)
       ↓ demonstrations
  MeanFlow Policy (78K params, FiLM-conditioned)
       ↓ pre-trained velocity field

Stage 2: RL Fine-Tuning (PPO)
  MeanFlow Policy → HFT Environment → PPO Agent
       ↓ fine-tuned
  Final Market-Making Agent
```

## Components

| Module | Description |
|--------|-------------|
| `simulator/` | Jump-diffusion + Hawkes process market simulator |
| `envs/` | OpenAI Gym-style HFT environment |
| `experts/` | Avellaneda-Stoikov, GLFT, GLFT-Drift expert policies |
| `models/` | MeanFlow (flow matching), NoisePolicy, FiLM layer |
| `agents/` | PPO agent with numpy MLP |
| `training/` | Stage 1 pre-trainer, Stage 2 fine-tuner |
| `evaluation/` | PnL, Sharpe Ratio, Maximum Drawdown |

## Quick Start

```bash
# Install
pip install -e .

# Run demo (verifies all components)
python scripts/demo.py

# Generate synthetic market data
python scripts/generate_data.py --steps 50000

# Train (pre-train + fine-tune)
python scripts/train.py --expert glft --pretrain-iters 500 --finetune-epochs 5

# Evaluate trained policy
python scripts/evaluate.py --checkpoint checkpoints/meanflow_params.json

# Run tests
python tests/run_all.py
```

## MeanFlow Policy

The core innovation is a **conditional flow-matching** policy:

- **Input**: Market observation (inventory, price, spread, volatility, order imbalance, Hawkes intensity)
- **Output**: Continuous trading action via Euler integration of learned velocity field
- **Conditioning**: FiLM (Feature-wise Linear Modulation) adapts the network to market state
- **Parameters**: ~78K (lightweight, pure numpy)

The flow matching objective:
```
L = E_{t,x₀,x₁} ||v_θ(x_t, t, cond) - (x₁ - x₀)||²
```

## Expert Strategies

- **Avellaneda-Stoikov**: Inventory-aware quoting with reservation price
- **GLFT**: Linear feature-based trading with heuristic weights
- **GLFT-Drift**: Extends GLFT with directional drift detection

## License

MIT
