#!/usr/bin/env python3
"""
Train a single TCM model with endpoint-constrained attention decoder.

Usage:
  python train_endpoints.py --rho 0.95 --sigma 0.05
  python train_endpoints.py --rho 0.90 --sigma 0.05 --epochs 4000 --seed 42
"""

import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    D_ITEM, D_CONTEXT, TEST_POSITIONS, HUMAN_ERRORS,
)
from src.train_endpoints import train_model_endpoints, evaluate_model_endpoints
from src.analysis import compute_summary_stats, compute_human_loss
from src.utils import log_dir, save_results


def main():
    parser = argparse.ArgumentParser(description='Train endpoint-constrained TCM')
    parser.add_argument('--rho', type=float, default=0.95)
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--d', type=int, default=D_CONTEXT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-interval', type=int, default=10,
                        help='Minimum A-B separation during training')
    args = parser.parse_args()

    print(f"Training endpoint-constrained model: ρ={args.rho:.2f}, σ={args.sigma:.2f}, d={args.d}, seed={args.seed}")

    encoder, decoder, train_mse = train_model_endpoints(
        args.rho, args.sigma, epochs=args.epochs, batch_size=args.batch,
        d_item=D_ITEM, d_context=args.d, seed=args.seed,
        min_interval=args.min_interval,
    )

    errors = evaluate_model_endpoints(encoder, decoder, args.sigma, d_item=D_ITEM)
    means, _, asymmetry = compute_summary_stats(errors)
    loss_val = compute_human_loss(means)
    eval_mae = float(np.mean([np.mean(np.abs(errors[p])) for p in TEST_POSITIONS]))

    print(f"  means: [{means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f}, {means[3]:.1f}]")
    print(f"  asymmetry: {asymmetry:+.1f}, loss: {loss_val:.1f}, MAE: {eval_mae:.2f}%, train MSE: {train_mse:.6f}")
    print(f"  Human:  {HUMAN_ERRORS}")

    ldir = log_dir(args.rho, args.sigma, args.d, args.seed, base='logs/endpoints')
    save_results(ldir, errors, {}, args.rho, args.sigma,
                 metadata={'epochs': args.epochs, 'loss': loss_val,
                           'asymmetry': asymmetry, 'd': args.d, 'seed': args.seed,
                           'train_mse': train_mse, 'eval_mae': eval_mae,
                           'model_type': 'constrained_attention',
                           'min_interval': args.min_interval})
    print(f"  logs: {ldir}")


if __name__ == '__main__':
    main()
