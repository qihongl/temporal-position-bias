#!/usr/bin/env python3
"""
Plot From Saved Data — No Retraining Required
================================================
Loads all completed runs from logs/ and regenerates figures.

Usage:
  python plot_from_data.py                              # plot all runs
  python plot_from_data.py --rho 0.95 --sigma 0.05      # single run
  python plot_from_data.py --sweep-only                  # sweep summary only
  python plot_from_data.py --all --regenerate            # force regenerate all
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_results, discover_runs, parse_run_path,
    log_dir, figure_dir, ensure_dir,
)
from src.analysis import compute_summary_stats, compute_human_loss
from src.plotting import (
    plot_error_curve, plot_error_distributions,
    plot_attention_weights, plot_sweep_summary,
)
from src.config import SEQ_TEST, TEST_POSITIONS, HUMAN_ERRORS


def plot_single_run(run_path, figs_root, force=False):
    """Plot all figures for a single saved run."""
    info = parse_run_path(run_path)
    fdir = figure_dir(info['rho'], info['sigma_m'], info['d'],
                       info['seed'], base=figs_root)

    ec_path = os.path.join(fdir, 'error_curve.png')
    if os.path.exists(ec_path) and not force:
        print(f"  Skipping (exists) — {fdir}")
        return

    errors, attn_data, meta = load_results(run_path)
    print(f"  Plotting ρ={info['rho']:.2f} σ={info['sigma_m']:.2f} seed={info['seed']}")

    plot_error_curve(errors, info['rho'], info['sigma_m'],
                     save_path=ec_path)
    plot_error_distributions(errors, info['rho'], info['sigma_m'],
                             save_path=os.path.join(fdir, 'error_distributions.png'))
    if attn_data:
        plot_attention_weights(attn_data, info['rho'], info['sigma_m'],
                               seq_test=SEQ_TEST,
                               save_path=os.path.join(fdir, 'attention_weights.png'))


def main():
    parser = argparse.ArgumentParser(description='Plot from saved TCM data')
    parser.add_argument('--rho', type=float, default=None)
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--sweep-only', action='store_true')
    parser.add_argument('--all', action='store_true',
                        help='Plot all discovered runs')
    parser.add_argument('--regenerate', action='store_true',
                        help='Force regenerate even if figures exist')
    parser.add_argument('--base', type=str, default='.')
    args = parser.parse_args()

    base = os.path.abspath(args.base)
    logs_root = os.path.join(base, 'logs')
    figs_root = os.path.join(base, 'figures')

    # ---- Sweep-only mode ----
    if args.sweep_only:
        runs = discover_runs(logs_root)
        print(f"Building sweep summary from {len(runs)} runs...")
        all_data = []
        for run_path in runs:
            errors, _, meta = load_results(run_path)
            means = [np.mean(errors[p]) for p in TEST_POSITIONS]
            loss = compute_human_loss(means)
            all_data.append({
                'rho': meta.get('rho', 0),
                'sigma_m': meta.get('sigma_m', 0),
                'errors_dict': errors,
                'loss': loss,
            })

        sweep_dir = os.path.join(figs_root, 'sweep_summary')
        ensure_dir(sweep_dir)
        save_path = os.path.join(sweep_dir, 'sweep_summary.png')
        plot_sweep_summary(all_data, save_path=save_path)
        print(f"Saved sweep summary to {save_path}")
        return

    # ---- Single run mode ----
    if args.rho is not None and args.sigma is not None:
        ldir = log_dir(args.rho, args.sigma, args.d, args.seed, base=logs_root)
        if not os.path.exists(os.path.join(ldir, 'errors.npz')):
            print(f"No saved data found at {ldir}")
            print(f"Run 'python main.py --rho {args.rho} --sigma {args.sigma}' first.")
            return
        plot_single_run(ldir, figs_root, force=args.regenerate)
        return

    # ---- Plot all mode ----
    if args.all:
        runs = discover_runs(logs_root)
        print(f"Plotting all {len(runs)} completed runs...")
        for run_path in runs:
            plot_single_run(run_path, figs_root, force=args.regenerate)

        # Also generate sweep summary
        all_data = []
        for run_path in runs:
            errors, _, meta = load_results(run_path)
            means = [np.mean(errors[p]) for p in TEST_POSITIONS]
            loss = compute_human_loss(means)
            all_data.append({
                'rho': meta.get('rho', 0),
                'sigma_m': meta.get('sigma_m', 0),
                'errors_dict': errors,
                'loss': loss,
            })
        sweep_dir = os.path.join(figs_root, 'sweep_summary')
        ensure_dir(sweep_dir)
        plot_sweep_summary(all_data, save_path=os.path.join(sweep_dir, 'sweep_summary.png'))
        print(f"\nDone. {len(runs)} runs plotted. Sweep summary at {sweep_dir}")
        return

    # ---- Default: list available runs ----
    runs = discover_runs(logs_root)
    print(f"Found {len(runs)} completed runs:\n")
    for run_path in runs[:20]:
        info = parse_run_path(run_path)
        print(f"  ρ={info['rho']:.2f}  σ_m={info['sigma_m']:.2f}  d={info['d']}  seed={info['seed']}")

    print(f"\nTo plot a specific run:")
    print(f"  python plot_from_data.py --rho 0.95 --sigma 0.05")
    print(f"To plot all runs:")
    print(f"  python plot_from_data.py --all")
    print(f"To generate sweep summary only:")
    print(f"  python plot_from_data.py --sweep-only")


if __name__ == '__main__':
    main()
