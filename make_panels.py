#!/usr/bin/env python3
"""
Generate 2×2 panel figure for each rho: signed error, asymmetry bars,
KDE distributions, CDF. Loads all seeds from logs/global/.
Usage: python make_panels.py [--log-dir logs/global] [--rho 0.95]
"""
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

from src.utils import load_results, discover_runs, parse_run_path

sns.set_context('talk')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
})

rhos = [0.6, 0.7, 0.8, 0.9, 0.95]
positions = [0.20, 0.40, 0.60, 0.80]
human = np.array([13.41, 7.57, -1.91, -7.36])
human_asym = (abs(human[0]) + abs(human[1]) - abs(human[2]) - abs(human[3])) / 2
rdylbu_4 = ['#D73027', '#F46D43', '#74ADD1', '#4575B4']


def load_seed_means(log_dir, rho=None, seed_range=(42, 46)):
    """Return per-seed mean signed errors and per-position trial data."""
    seed_means = {pf: [] for pf in positions}
    all_errors = {pf: [] for pf in positions}
    for run_path in discover_runs(log_dir):
        info = parse_run_path(run_path)
        if rho is not None and abs(info['rho'] - rho) > 0.005:
            continue
        if not (seed_range[0] <= info['seed'] <= seed_range[1]):
            continue
        errors, _, _ = load_results(run_path)
        for pf in positions:
            seed_means[pf].append(np.mean(errors[pf]))
            all_errors[pf].extend(errors[pf].tolist())
    return seed_means, all_errors


def make_panels(log_dir, rho, suffix='', rnn_errors=None, seed_range=(42, 46)):
    seed_means, all_errors = load_seed_means(log_dir, rho=rho if rho is not None else None, seed_range=seed_range)
    if not seed_means[0.20]:
        print(f'{log_dir} (rho={rho}): NO DATA')
        return

    means = [np.mean(seed_means[pf]) for pf in positions]
    sems = [np.std(seed_means[pf], ddof=1) / np.sqrt(len(seed_means[pf])) for pf in positions]
    asym = (abs(means[0]) + abs(means[1]) - abs(means[2]) - abs(means[3])) / 2

    # Per-seed early/late for SEM propagation
    seed_early = [abs(np.mean(seed_means[0.20])) for _ in seed_means[0.20]]  # placeholder
    n_seeds = len(seed_means[0.20])
    seed_vals = []
    for si in range(n_seeds):
        vals = [seed_means[pf][si] for pf in positions]
        seed_vals.append(vals)
    seed_early = [(abs(v[0]) + abs(v[1])) / 2 for v in seed_vals]
    seed_late = [(abs(v[2]) + abs(v[3])) / 2 for v in seed_vals]
    early_mean = np.mean(seed_early)
    late_mean = np.mean(seed_late)
    early_sem = np.std(seed_early, ddof=1) / np.sqrt(n_seeds)
    late_sem = np.std(seed_late, ddof=1) / np.sqrt(n_seeds)
    human_early = np.mean([abs(human[i]) for i in [0, 1]])
    human_late = np.mean([abs(human[i]) for i in [2, 3]])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # A: Signed error
    ax = axes[0]
    err_band = [3 * s for s in sems]
    ax.fill_between([20, 40, 60, 80],
                    [m - e for m, e in zip(means, err_band)],
                    [m + e for m, e in zip(means, err_band)],
                    color='#E6550D', alpha=0.15)
    ax.errorbar([20, 40, 60, 80], means, yerr=sems, fmt='o-', color='#E6550D',
                lw=1.5, ms=5, capsize=3, label='TCM')
    if rnn_errors is not None:
        rnn_means = [np.mean(rnn_errors[pf]) for pf in positions]
        rnn_sems = [np.std(rnn_errors[pf])/np.sqrt(len(rnn_errors[pf])) for pf in positions]
        ax.errorbar([20, 40, 60, 80], rnn_means, yerr=rnn_sems, fmt='o-', color='gray',
                    lw=1.5, ms=5, capsize=3, label='RNN')
    ax.plot([20, 40, 60, 80], human, 's-', color='black', ms=5, lw=1.5,
            label='Human')
    ax.set_xlabel('True Position (%)'); ax.set_ylabel('Signed Error (%)')
    ax.legend(); ax.set_xticks([20, 40, 60, 80])

    # B: Asymmetry bars
    ax = axes[1]
    ax.bar([0, 1], [early_mean, late_mean], color=['#D73027', '#4575B4'],
            edgecolor='white', lw=1, width=0.5)
    ax.errorbar([0, 1], [early_mean, late_mean],
                yerr=[early_sem, late_sem],
                fmt='none', color='black', capsize=5, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['20% + 40%', '60% + 80%'])

    # C: KDE distributions
    ax = axes[2]
    x_all = np.concatenate([all_errors[pf] for pf in positions])
    xs = np.linspace(x_all.min() - 5, x_all.max() + 5, 200)
    for i, pf in enumerate(positions):
        kde = gaussian_kde(np.array(all_errors[pf]))
        ax.fill_between(xs, kde(xs), alpha=0.3, color=rdylbu_4[i])
        ax.plot(xs, kde(xs), color=rdylbu_4[i], lw=1.5, label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('Density')

    # D: CDF
    ax = axes[3]
    for i, pf in enumerate(positions):
        data = np.sort(all_errors[pf])
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf, color=rdylbu_4[i], lw=1.5, label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('CDF')
    ax.legend(loc='lower right')

    fig.tight_layout()
    label = f'rho{rho:.2f}' if rho is not None else suffix
    out_path = f'figures/tcm_{label}_panels.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'{label}: asym={asym:+.1f}, saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='logs/global')
    parser.add_argument('--rho', type=float, default=None)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()

    rnn_seed_means, rnn_errors = load_seed_means('logs/rnn', rho=0.0, seed_range=(42, 46))

    if args.rho is not None:
        sr = (42, 61) if args.rho == 0.95 else (42, 46)
        make_panels(args.log_dir, args.rho, rnn_errors=rnn_errors if args.rho != 0.0 else None, seed_range=sr)
    else:
        for rho in rhos:
            sr = (42, 61) if rho == 0.95 else (42, 46)
            make_panels(args.log_dir, rho, args.suffix, rnn_errors=rnn_errors, seed_range=sr)


if __name__ == '__main__':
    main()
