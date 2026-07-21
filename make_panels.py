#!/usr/bin/env python3
"""
Generate 4-panel figure for each rho: signed error, asymmetry bars,
KDE distributions, CDF. Loads all seeds from logs/global/.
Usage: python make_panels.py [--log-dir logs/global] [--rho 0.95]
       [--human-mode mu|signed]
"""
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde, vonmises
from scipy.optimize import minimize

from src.utils import load_results, discover_runs, parse_run_path

sns.set_context('talk')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
})

rhos = [0.6, 0.7, 0.8, 0.9, 0.95]
positions = [0.20, 0.40, 0.60, 0.80]
rdylbu_4 = ['#D73027', '#F46D43', '#74ADD1', '#4575B4']

_HUMAN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'human-data', 'temporal-bias-human-analysis',
    'data', 'data', 'behavioral_18sub')

_MU_XLSX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'human-data', 'temporal-bias-human-analysis',
    'data', 'processed_memdata', 'memresults.xlsx')


def load_human_signed_error(data_dir=_HUMAN_DATA_DIR):
    """Load 18-subject signed error (%) — subject means per td, pooled across intervals."""
    if not os.path.exists(data_dir):
        return None
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx')])
    dfs = []
    for f in files:
        pid = f.split('_')[0]
        df = pd.read_excel(os.path.join(data_dir, f), header=0)
        df['subject'] = pid
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)
    subj_means = all_data.groupby(['subject', 'td'])['ERROR'].mean().reset_index()
    means = subj_means.groupby('td')['ERROR'].mean()
    sems = subj_means.groupby('td')['ERROR'].sem()
    return {td: {'mean': means[td], 'sem': sems[td]} for td in positions}


def load_human_mu(mu_path=_MU_XLSX_PATH):
    """Load 16-subject MemToolbox μ — subject estimates per td, pooled across intervals."""
    if not os.path.exists(mu_path):
        return None
    mu_df = pd.read_excel(mu_path, sheet_name='Sheet1')
    means = mu_df.groupby('td')['mu'].mean()
    sems = mu_df.groupby('td')['mu'].sem()
    return {td: {'mean': means[td], 'sem': sems[td]} for td in positions}


def fit_memtoolbox_mu(errors_pct):
    """Fit von Mises mixture model (Orientation(WithBias(StandardMixtureModel)))
    via MLE — returns μ in degrees. Errors are clipped to [-80, 80] then asind-transformed."""
    errors_pct = np.clip(errors_pct, -80, 80)
    theta = np.radians(np.degrees(np.arcsin(errors_pct / 80)))

    def nll(params):
        mu, log_kappa, logit_g = params
        kappa = np.exp(log_kappa)
        g = 1.0 / (1.0 + np.exp(-logit_g))
        g = np.clip(g, 1e-9, 1 - 1e-9)
        like = (1 - g) * vonmises.pdf(theta, kappa, loc=mu) + g / (2 * np.pi)
        return -np.sum(np.log(np.maximum(like, 1e-300)))

    res = minimize(nll, [0.0, np.log(10), 0.0],
                   bounds=[(-np.pi / 2, np.pi / 2),
                           (np.log(0.1), np.log(1e6)),
                           (-20, 20)],
                   method='L-BFGS-B', options={'maxiter': 1000})
    return np.degrees(res.x[0])


def load_model_mu(log_dir, rho=None, seed_range=(42, 46)):
    """Fit MemToolbox μ per seed per td from model trial-level errors."""
    seed_mu = {pf: [] for pf in positions}
    for run_path in discover_runs(log_dir):
        info = parse_run_path(run_path)
        if rho is not None and abs(info['rho'] - rho) > 0.005:
            continue
        if not (seed_range[0] <= info['seed'] <= seed_range[1]):
            continue
        errors, _, _ = load_results(run_path)
        for pf in positions:
            mu_deg = fit_memtoolbox_mu(errors[pf])
            seed_mu[pf].append(mu_deg)
    return seed_mu


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


def make_panels(log_dir, rho, suffix='', rnn_errors=None, seed_range=(42, 46),
                human_mode='mu'):
    seed_means, all_errors = load_seed_means(log_dir, rho=rho if rho is not None else None, seed_range=seed_range)
    if not seed_means[0.20]:
        print(f'{log_dir} (rho={rho}): NO DATA')
        return

    # Human data
    if human_mode == 'signed':
        human_data = load_human_signed_error()
    else:
        human_data = load_human_mu()
    human_means = [human_data[pf]['mean'] for pf in positions] if human_data else [np.nan]*4
    human_sems = [human_data[pf]['sem'] for pf in positions] if human_data else [np.nan]*4
    human_label = 'Human' if human_mode == 'signed' else 'Human (μ)'

    # Model data for Panel A and B
    if human_mode == 'mu':
        model_mu = load_model_mu(log_dir, rho=rho if rho is not None else None, seed_range=seed_range)
        means = [np.mean(model_mu[pf]) for pf in positions]
        sems = [np.std(model_mu[pf], ddof=1) / np.sqrt(max(len(model_mu[pf]), 1)) for pf in positions]
        ylabel = 'μ'
    else:
        means = [np.mean(seed_means[pf]) for pf in positions]
        sems = [np.std(seed_means[pf], ddof=1) / np.sqrt(len(seed_means[pf])) for pf in positions]
        ylabel = 'Signed Error (%)'

    asym = (abs(means[0]) + abs(means[1]) - abs(means[2]) - abs(means[3])) / 2

    # Per-seed early/late
    if human_mode == 'mu':
        n_seeds = len(model_mu[0.20])
        seed_early = [(abs(model_mu[0.20][i]) + abs(model_mu[0.40][i])) / 2 for i in range(n_seeds)]
        seed_late = [(abs(model_mu[0.60][i]) + abs(model_mu[0.80][i])) / 2 for i in range(n_seeds)]
    else:
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
    human_early = np.mean([abs(human_means[i]) for i in [0, 1]])
    human_late = np.mean([abs(human_means[i]) for i in [2, 3]])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # A: Signed error
    ax = axes[0]
    ax.fill_between([20, 40, 60, 80],
                    [m - s for m, s in zip(means, sems)],
                    [m + s for m, s in zip(means, sems)],
                    color='#E6550D', alpha=0.15)
    ax.errorbar([20, 40, 60, 80], means, yerr=sems, fmt='o-', color='#E6550D',
                lw=1.5, ms=5, capsize=3, label='TCM')
    if rnn_errors is not None:
        rnn_means = [np.mean(rnn_errors[pf]) for pf in positions]
        rnn_sems = [np.std(rnn_errors[pf])/np.sqrt(len(rnn_errors[pf])) for pf in positions]
        ax.errorbar([20, 40, 60, 80], rnn_means, yerr=rnn_sems, fmt='o-', color='gray',
                    lw=1.5, ms=5, capsize=3, label='RNN')
    if human_data:
        ax.fill_between([20, 40, 60, 80],
                        [m - s for m, s in zip(human_means, human_sems)],
                        [m + s for m, s in zip(human_means, human_sems)],
                        color='black', alpha=0.10)
        ax.errorbar([20, 40, 60, 80], human_means, yerr=human_sems, fmt='s-',
                    color='black', ms=5, lw=1.5, capsize=3, label=human_label)
    ax.set_xlabel('True Position (%)'); ax.set_ylabel(ylabel)
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
    out_path = f'figures/tcm_{label}_panels_{human_mode}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'{label} [{human_mode}]: asym={asym:+.1f}, saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='logs/global')
    parser.add_argument('--rho', type=float, default=None)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--human-mode', default='mu', choices=['mu', 'signed'],
                        help='mu = MemToolbox bias values; signed = raw signed error %%')
    args = parser.parse_args()

    rnn_seed_means, rnn_errors = load_seed_means('logs/rnn', rho=0.0, seed_range=(42, 46))

    if args.rho is not None:
        sr = (42, 61) if args.rho == 0.95 else (42, 46)
        make_panels(args.log_dir, args.rho, rnn_errors=rnn_errors if args.rho != 0.0 else None,
                    seed_range=sr, human_mode=args.human_mode)
    else:
        for rho in rhos:
            sr = (42, 61) if rho == 0.95 else (42, 46)
            make_panels(args.log_dir, rho, args.suffix, rnn_errors=rnn_errors,
                        seed_range=sr, human_mode=args.human_mode)


if __name__ == '__main__':
    main()
