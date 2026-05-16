#!/usr/bin/env python3
"""CLI: Generate synthetic market data."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finflowrl.data.generate import generate_market_data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic market data")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/market.npz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    data = generate_market_data(n_steps=args.steps, seed=args.seed, save_path=args.output)
    print(f"[Data] Generated {args.steps} steps -> {args.output}")
    print(f"       Mid-price range: [{data['mid_price'].min():.2f}, {data['mid_price'].max():.2f}]")
    print(f"       Avg spread: {data['spread'].mean():.6f}")


if __name__ == "__main__":
    main()
