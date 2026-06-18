#!/usr/bin/env python3
"""
TCM Main — Train, Evaluate, and Plot a Single Model
======================================================
Usage:
  python main.py --rho 0.95 --sigma 0.10 --seed 42
  python main.py --rho 0.90 --sigma 0.05 --seed 1 --epochs 2000
  python main.py --plot-only  # plot from existing logs

All results are saved to logs/ and figures/ with nested directory
structure reflecting model parameters.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys

# Add src to path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    D_ITEM, D_CONTEXT, EPOCHS, SEQ_TEST,
    TEST_POSITIONS, HUMAN_ERRORS,
)
from src.train import train_model, evaluate_model, get_attention_weights
from src.analysis import compute_summary_stats, compute_human_loss
from src.utils import log_dir, figure_dir, save_results, load_results, discover_runs
from src.plotting import (
    plot_error_curve, plot_error_distributions,
    plot_attention_weights, plot_sweep_summary,
)


def main():
    parser = argparse.ArgumentParser(description='TCM Temporal Position Model')
    parser.add_argument('--rho', type=float, default=0.95)
    parser.add_argument('--sigma', type=float, default=0.10)
    parser.add_argument('--d', type=int, default=D_CONTEXT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--plot-only', action='store_true',
                        help='Plot from saved data only, no training')
    parser.add_argument('--sweep-plot', action='store_true',
                        help='Generate sweep summary from all saved runs')
    parser.add_argument('--base', type=str, default='.', help='Base directory')
    args = parser.parse_args()

    # ---- Sweep summary mode ----
    if args.sweep_plot:
        logs_dir = os.path.join(args.base, 'logs')
        runs = discover_runs(logs_dir)
        print(f"Found {len(runs)} completed runs.")

        all_data = []
        for run_path in runs:
            errors, _, meta = load_results(run_path)
            info = {
                'rho': meta.get('rho', 0),
                'sigma_m': meta.get('sigma_m', 0),
                'errors_dict': errors,
            }
            # Compute loss
            means = [np.mean(errors[p]) for p in TEST_POSITIONS]
            info['loss'] = compute_human_loss(means)
            all_data.append(info)

        sweep_dir = os.path.join(args.base, 'figures', 'sweep_summary')
        os.makedirs(sweep_dir, exist_ok=True)
        save_path = os.path.join(sweep_dir, 'sweep_summary.png')
        plot_sweep_summary(all_data, save_path=save_path)
        print(f"Saved sweep summary to {save_path}")
        return

    # ---- Plot-only mode ----
    if args.plot_only:
        ldir = log_dir(args.rho, args.sigma, args.d, args.seed, base=os.path.join(args.base, 'logs'))
        fdir = figure_dir(args.rho, args.sigma, args.d, args.seed, base=os.path.join(args.base, 'figures'))

        errors, attn_data, meta = load_results(ldir)
        if not meta:
            meta = {'rho': args.rho, 'sigma_m': args.sigma}
        print(f"Loaded results from {ldir}")

        means, _, asymmetry = compute_summary_stats(errors)
        print(f"  Means: [{means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f}, {means[3]:.1f}]")
        print(f"  Asymmetry: {asymmetry:+.1f}")

        plot_error_curve(errors, meta.get('rho', args.rho), meta.get('sigma_m', args.sigma),
                         save_path=os.path.join(fdir, 'error_curve.png'))
        plot_error_distributions(errors, meta.get('rho', args.rho), meta.get('sigma_m', args.sigma),
                                 save_path=os.path.join(fdir, 'error_distributions.png'))
        if attn_data:
            plot_attention_weights(attn_data, meta.get('rho', args.rho), meta.get('sigma_m', args.sigma),
                                   seq_test=SEQ_TEST,
                                   save_path=os.path.join(fdir, 'attention_weights.png'))

        print(f"Figures saved to {fdir}")
        return

    # ---- Train + Eval mode ----
    print(f"Training: ρ={args.rho}, σ_m={args.sigma}, d={args.d}, seed={args.seed}")

    encoder, decoder = train_model(args.rho, args.sigma, epochs=args.epochs,
                                   d_item=D_ITEM, d_context=args.d, seed=args.seed)

    errors = evaluate_model(encoder, decoder, args.sigma, d_item=D_ITEM)
    attn = get_attention_weights(encoder, decoder, args.sigma, d_item=D_ITEM)
    means, _, asymmetry = compute_summary_stats(errors)

    loss = compute_human_loss(means)
    print(f"  Means: [{means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f}, {means[3]:.1f}]")
    print(f"  Asymmetry: {asymmetry:+.1f}, Loss: {loss:.1f}")

    # Save
    ldir = log_dir(args.rho, args.sigma, args.d, args.seed, base=os.path.join(args.base, 'logs'))
    save_results(ldir, errors, attn, args.rho, args.sigma,
                 metadata={'epochs': args.epochs, 'loss': loss, 'asymmetry': asymmetry})
    print(f"  Results saved to {ldir}")

    # Plot
    fdir = figure_dir(args.rho, args.sigma, args.d, args.seed, base=os.path.join(args.base, 'figures'))
    plot_error_curve(errors, args.rho, args.sigma,
                     save_path=os.path.join(fdir, 'error_curve.png'))
    plot_error_distributions(errors, args.rho, args.sigma,
                             save_path=os.path.join(fdir, 'error_distributions.png'))
    plot_attention_weights(attn, args.rho, args.sigma, seq_test=SEQ_TEST,
                           save_path=os.path.join(fdir, 'attention_weights.png'))

    print(f"  Figures saved to {fdir}")


if __name__ == '__main__':
    main()
