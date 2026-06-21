#!/usr/bin/env python3
"""
TCM Parameter Sweep Runner
===========================
Grid-search over rho and sigma_measurement.
Each model's results are saved to nested logs/ and figures/ directories.

Usage:
  python sweep.py                          # default sweep
  python sweep.py --rhos 0.8,0.9,0.95      # custom rhos
  python sweep.py --sigmas 0.05,0.10       # custom sigmas
  python sweep.py --epochs 1000 --d 32      # faster sweep
"""

import argparse
import numpy as np
import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    D_ITEM, D_CONTEXT, SWEEP_RHOS, SWEEP_SIGMAS, SWEEP_EPOCHS, SWEEP_BATCH,
    SEQ_TEST, TEST_POSITIONS, HUMAN_ERRORS, USE_POSITION_TEMPLATE,
)
from src.train import train_model, evaluate_model, get_attention_weights
from src.analysis import compute_summary_stats, compute_human_loss
from src.utils import log_dir, figure_dir, save_results, load_results, discover_runs
from src.plotting import plot_error_curve, plot_error_distributions, plot_attention_weights


def main():
    parser = argparse.ArgumentParser(description='TCM Parameter Sweep')
    parser.add_argument('--rhos', type=str, default=','.join(str(r) for r in SWEEP_RHOS),
                        help='Comma-separated rho values')
    parser.add_argument('--sigmas', type=str, default=','.join(str(s) for s in SWEEP_SIGMAS),
                        help='Comma-separated sigma values')
    parser.add_argument('--epochs', type=int, default=SWEEP_EPOCHS)
    parser.add_argument('--batch', type=int, default=SWEEP_BATCH)
    parser.add_argument('--d', type=int, default=D_CONTEXT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for multi-seed runs (overrides --seed)')
    parser.add_argument('--base', type=str, default='.')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation (faster)')
    parser.add_argument('--no-template', action='store_true',
                        help='Remove (i/L) position template; decoder learns'
                             ' position from retrieved context mixture via MLP')
    parser.add_argument('--log-name', type=str, default=None,
                        help='Custom log directory name (e.g., "replication")')
    args = parser.parse_args()

    rhos = [float(r) for r in args.rhos.split(',')]
    sigmas = [float(s) for s in args.sigmas.split(',')]

    # Handle seeds: either single --seed or multi-seed list
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seeds = [args.seed]

    total = len(rhos) * len(sigmas) * len(seeds)
    print("=" * 60)
    print(f"TCM Parameter Sweep: {total} models")
    print(f"  ρ: {rhos}")
    print(f"  σ_m: {sigmas}")
    print(f"  seeds: {seeds}")
    print(f"  d: {args.d}, epochs: {args.epochs}")
    print("=" * 60)

    loss_grid = np.zeros((len(rhos), len(sigmas)))
    means_grid = np.zeros((len(rhos), len(sigmas), 4))

    base_dir = os.path.abspath(args.base)
    use_template = not args.no_template
    if args.log_name:
        suffix = f'_{args.log_name}'
    else:
        suffix = '' if use_template else '_notmpl'
    logs_root = os.path.join(base_dir, f'logs{suffix}')
    figs_root = os.path.join(base_dir, f'figures{suffix}')

    print(f"  Decoder: {'linear template' if use_template else 'learned pos_fn (no template)'}")
    print("=" * 60)

    for i, rho in enumerate(rhos):
        for j, sm in enumerate(sigmas):
            for seed in seeds:
                counter = i * len(sigmas) * len(seeds) + j * len(seeds) + seeds.index(seed) + 1
                label = f"ρ={rho:.2f} σ_m={sm:.2f} seed={seed}"
                print(f"\n[{counter}/{total}] {label}")

                # Check if already done
                ldir = log_dir(rho, sm, args.d, seed, base=logs_root)
                if os.path.exists(os.path.join(ldir, 'errors.npz')):
                    print(f"  → Already completed, skipping")
                    continue

                # Train
                encoder, decoder, train_mse = train_model(
                    rho, sm, epochs=args.epochs, batch_size=args.batch,
                    d_item=D_ITEM, d_context=args.d, seed=seed,
                    use_position_template=use_template,
                )

                # Evaluate
                errors = evaluate_model(encoder, decoder, sm, d_item=D_ITEM)
                attn = get_attention_weights(encoder, decoder, sm, d_item=D_ITEM)
                means, _, asymmetry = compute_summary_stats(errors)
                loss_val = compute_human_loss(means)
                eval_mae = float(np.mean([np.mean(np.abs(errors[p])) for p in TEST_POSITIONS]))

                # Store best seed's loss for the grid (first seed, or best)
                if seed == seeds[0]:
                    loss_grid[i, j] = loss_val
                    means_grid[i, j] = means

                print(f"  → means: [{means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f}, {means[3]:.1f}]")
                print(f"  → asymmetry: {asymmetry:+.1f}, loss: {loss_val:.1f}, MAE: {eval_mae:.2f}%, train MSE: {train_mse:.6f}")

                # Save logs
                save_results(ldir, errors, attn, rho, sm,
                             metadata={'epochs': args.epochs, 'loss': loss_val,
                                       'asymmetry': asymmetry, 'd': args.d, 'seed': seed,
                                       'train_mse': train_mse, 'eval_mae': eval_mae,
                                       'use_position_template': use_template})
                print(f"  → logs: {ldir}")

                # Save figures (only for first seed, avoid bloat)
                if not args.no_figures and seed == seeds[0]:
                    fdir = figure_dir(rho, sm, args.d, seed, base=figs_root)
                    plot_error_curve(errors, rho, sm,
                                     save_path=os.path.join(fdir, 'error_curve.png'))
                    plot_error_distributions(errors, rho, sm,
                                             save_path=os.path.join(fdir, 'error_distributions.png'))
                    plot_attention_weights(attn, rho, sm, seq_test=SEQ_TEST,
                                           save_path=os.path.join(fdir, 'attention_weights.png'))
                    print(f"  → figures: {fdir}")

                # Free memory
                del encoder, decoder; gc.collect()

    # Summary
    best_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    best_rho, best_sigma = rhos[best_idx[0]], sigmas[best_idx[1]]
    best_loss = loss_grid[best_idx[0], best_idx[1]]
    best_means = means_grid[best_idx[0], best_idx[1]]

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"  Best: ρ={best_rho:.2f}, σ_m={best_sigma:.2f}, loss={best_loss:.1f}")
    print(f"  Model: [{best_means[0]:.1f}, {best_means[1]:.1f}, {best_means[2]:.1f}, {best_means[3]:.1f}]")
    print(f"  Human: {HUMAN_ERRORS}")

    # Print loss grid
    print(f"\n{'ρ\\σ_m':>7}", end="")
    for s in sigmas:
        print(f"{s:>8.2f}", end="")
    print(f"\n{'-'*(7+8*len(sigmas))}")
    for i, rho in enumerate(rhos):
        print(f"{rho:>7.2f}", end="")
        for j in range(len(sigmas)):
            print(f"{loss_grid[i,j]:>8.1f}", end="")
        print()

    # Generate sweep summary figure from saved data
    print(f"\nGenerating sweep summary figure...")
    runs = discover_runs(logs_root)
    all_data = []
    for run_path in runs:
        errors, _, meta = load_results(run_path)
        all_data.append({
            'rho': meta.get('rho', 0),
            'sigma_m': meta.get('sigma_m', 0),
            'errors_dict': errors,
            'loss': meta.get('loss', 0),
        })

    from src.plotting import plot_sweep_summary
    sweep_dir = os.path.join(figs_root, 'sweep_summary')
    os.makedirs(sweep_dir, exist_ok=True)
    plot_sweep_summary(all_data, save_path=os.path.join(sweep_dir, 'sweep_summary.png'))
    print(f"  Sweep summary saved to {sweep_dir}")


if __name__ == '__main__':
    main()
