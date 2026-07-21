#!/usr/bin/env python3
"""
Generate all figures from fixed-target results.
Reads from logs/fixed_target/ and logs/rnn_fixed/.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

from src.utils import load_results, discover_runs, parse_run_path
from src.config import TEST_POSITIONS, HUMAN_ERRORS
from src.analysis import fit_memtoolbox_mu, load_model_mu, load_seed_means
from src.human_data import load_human_mu, load_human_signed_error

sns.set_context('talk')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
})

POSITIONS = list(TEST_POSITIONS)
PX = [20, 40, 60, 80]
RHYLOS = [0.60, 0.70, 0.80, 0.90, 0.95]
rdylbu_4 = ['#D73027', '#F46D43', '#74ADD1', '#4575B4']
viridis = plt.cm.viridis
rho_colors = {r: viridis(i / (len(RHYLOS) - 1)) for i, r in enumerate(RHYLOS)}

LOGS_TCM = os.path.join(ROOT, 'logs', 'fixed_target')
LOGS_RNN = os.path.join(ROOT, 'logs', 'rnn_fixed')
OUT = os.path.join(ROOT, 'figures')


# ============================================================
# Figure 1: Panels for each rho (mu mode)
# ============================================================
human_mu_data = load_human_mu()
human_mu_means = [human_mu_data[pf]['mean'] for pf in POSITIONS] if human_mu_data else [np.nan]*4
human_mu_sems  = [human_mu_data[pf]['sem'] for pf in POSITIONS] if human_mu_data else [np.nan]*4

# RNN baseline
rnn_model_mu = load_model_mu(LOGS_RNN, 0.0)
rnn_mu_means = [np.mean(rnn_model_mu[pf]) for pf in POSITIONS] if rnn_model_mu[0.20] else None
rnn_mu_sems = [np.std(rnn_model_mu[pf], ddof=1)/np.sqrt(len(rnn_model_mu[pf]))
               for pf in POSITIONS] if rnn_model_mu[0.20] else None

for rho in RHYLOS:
    print(f"\nPanels for ρ={rho:.2f} (mu mode)...")
    seed_means, all_errors = load_seed_means(LOGS_TCM, rho)
    model_mu = load_model_mu(LOGS_TCM, rho)

    if not seed_means[0.20]:
        print(f"  NO DATA")
        continue

    n_seeds = len(seed_means[0.20])
    mu_means = [np.mean(model_mu[pf]) for pf in POSITIONS]
    mu_sems = [np.std(model_mu[pf], ddof=1)/np.sqrt(max(len(model_mu[pf]),1))
               for pf in POSITIONS]
    asym = (abs(mu_means[0]) + abs(mu_means[1]) -
            abs(mu_means[2]) - abs(mu_means[3])) / 2

    seed_early = [(abs(model_mu[0.20][i]) + abs(model_mu[0.40][i])) / 2
                  for i in range(n_seeds)]
    seed_late = [(abs(model_mu[0.60][i]) + abs(model_mu[0.80][i])) / 2
                 for i in range(n_seeds)]
    early_mean = np.mean(seed_early); late_mean = np.mean(seed_late)
    early_sem = np.std(seed_early, ddof=1)/np.sqrt(n_seeds)
    late_sem = np.std(seed_late, ddof=1)/np.sqrt(n_seeds)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # A: μ error curve
    ax = axes[0]
    ax.fill_between(PX,
                    [m - s for m, s in zip(mu_means, mu_sems)],
                    [m + s for m, s in zip(mu_means, mu_sems)],
                    color='#E6550D', alpha=0.15)
    ax.errorbar(PX, mu_means, yerr=mu_sems, fmt='o-', color='#E6550D',
                lw=1.5, ms=5, capsize=3, label=f'TCM (n={n_seeds})')
    if rnn_mu_means is not None:
        rnn_sems = [np.std(rnn_model_mu[pf], ddof=1)/np.sqrt(len(rnn_model_mu[pf]))
                    for pf in POSITIONS]
        ax.errorbar(PX, rnn_mu_means, yerr=rnn_sems, fmt='o-', color='gray',
                    lw=1.5, ms=5, capsize=3, label='RNN')
    if human_mu_data:
        ax.fill_between(PX,
                        [m - s for m, s in zip(human_mu_means, human_mu_sems)],
                        [m + s for m, s in zip(human_mu_means, human_mu_sems)],
                        color='black', alpha=0.10)
        ax.errorbar(PX, human_mu_means, yerr=human_mu_sems, fmt='s-',
                    color='black', ms=5, lw=1.5, capsize=3, label='Human (μ)')
    ax.set_xlabel('True Position (%)'); ax.set_ylabel('μ')
    ax.legend(fontsize=8); ax.set_xticks(PX)

    # B: Asymmetry bars
    ax = axes[1]
    ax.bar([0, 1], [early_mean, late_mean], color=['#D73027', '#4575B4'],
            edgecolor='white', lw=1, width=0.5)
    ax.errorbar([0, 1], [early_mean, late_mean],
                yerr=[early_sem, late_sem],
                fmt='none', color='black', capsize=5, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['20% + 40%', '60% + 80%'])

    # C: KDE distributions (signed error)
    ax = axes[2]
    x_all = np.concatenate([all_errors[pf] for pf in POSITIONS])
    xs = np.linspace(x_all.min() - 5, x_all.max() + 5, 200)
    for i, pf in enumerate(POSITIONS):
        kde = gaussian_kde(np.array(all_errors[pf]))
        ax.fill_between(xs, kde(xs), alpha=0.3, color=rdylbu_4[i])
        ax.plot(xs, kde(xs), color=rdylbu_4[i], lw=1.5,
                label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('Density')

    # D: CDF
    ax = axes[3]
    for i, pf in enumerate(POSITIONS):
        data = np.sort(all_errors[pf])
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf, color=rdylbu_4[i], lw=1.5,
                label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('CDF')
    ax.legend(loc='lower right')

    fig.tight_layout()
    path = os.path.join(OUT, f'tcm_rho{rho:.2f}_panels_mu.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}  (asym={asym:+.1f}, n={n_seeds})")


# ============================================================
# Figure 2: Panels for each rho (signed error mode)
# ============================================================
human_se_data = load_human_signed_error()
human_se_means = [human_se_data[pf]['mean'] for pf in POSITIONS] if human_se_data else [np.nan]*4
human_se_sems  = [human_se_data[pf]['sem'] for pf in POSITIONS] if human_se_data else [np.nan]*4

rnn_se, _ = load_seed_means(LOGS_RNN, 0.0)
rnn_se_means = [np.mean(rnn_se[pf]) for pf in POSITIONS] if rnn_se and rnn_se[0.20] else None
rnn_se_sems = [np.std(rnn_se[pf], ddof=1)/np.sqrt(len(rnn_se[pf]))
               for pf in POSITIONS] if rnn_se and rnn_se[0.20] else None

for rho in RHYLOS:
    print(f"\nPanels for ρ={rho:.2f} (signed mode)...")
    seed_means, all_errors = load_seed_means(LOGS_TCM, rho)
    if not seed_means[0.20]:
        print(f"  NO DATA")
        continue

    n_seeds = len(seed_means[0.20])
    se_means = [np.mean(seed_means[pf]) for pf in POSITIONS]
    se_sems = [np.std(seed_means[pf], ddof=1)/np.sqrt(len(seed_means[pf]))
               for pf in POSITIONS]
    asym = (abs(se_means[0]) + abs(se_means[1]) -
            abs(se_means[2]) - abs(se_means[3])) / 2

    seed_early = [(abs(seed_means[0.20][i]) + abs(seed_means[0.40][i])) / 2
                  for i in range(n_seeds)]
    seed_late = [(abs(seed_means[0.60][i]) + abs(seed_means[0.80][i])) / 2
                 for i in range(n_seeds)]
    early_mean = np.mean(seed_early); late_mean = np.mean(seed_late)
    early_sem = np.std(seed_early, ddof=1)/np.sqrt(n_seeds)
    late_sem = np.std(seed_late, ddof=1)/np.sqrt(n_seeds)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # A
    ax = axes[0]
    ax.fill_between(PX,
                    [m - s for m, s in zip(se_means, se_sems)],
                    [m + s for m, s in zip(se_means, se_sems)],
                    color='#E6550D', alpha=0.15)
    ax.errorbar(PX, se_means, yerr=se_sems, fmt='o-', color='#E6550D',
                lw=1.5, ms=5, capsize=3, label=f'TCM (n={n_seeds})')
    if rnn_se_means is not None:
        ax.errorbar(PX, rnn_se_means, yerr=rnn_se_sems, fmt='o-', color='gray',
                    lw=1.5, ms=5, capsize=3, label='RNN')
    if human_se_data:
        ax.fill_between(PX,
                        [m - s for m, s in zip(human_se_means, human_se_sems)],
                        [m + s for m, s in zip(human_se_means, human_se_sems)],
                        color='black', alpha=0.10)
        ax.errorbar(PX, human_se_means, yerr=human_se_sems, fmt='s-',
                    color='black', ms=5, lw=1.5, capsize=3, label='Human')
    ax.set_xlabel('True Position (%)'); ax.set_ylabel('Signed Error (%)')
    ax.legend(fontsize=8); ax.set_xticks(PX)

    # B
    ax = axes[1]
    ax.bar([0, 1], [early_mean, late_mean], color=['#D73027', '#4575B4'],
            edgecolor='white', lw=1, width=0.5)
    ax.errorbar([0, 1], [early_mean, late_mean],
                yerr=[early_sem, late_sem], fmt='none', color='black', capsize=5, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['20% + 40%', '60% + 80%'])

    # C
    ax = axes[2]
    x_all = np.concatenate([all_errors[pf] for pf in POSITIONS])
    xs = np.linspace(x_all.min() - 5, x_all.max() + 5, 200)
    for i, pf in enumerate(POSITIONS):
        kde = gaussian_kde(np.array(all_errors[pf]))
        ax.fill_between(xs, kde(xs), alpha=0.3, color=rdylbu_4[i])
        ax.plot(xs, kde(xs), color=rdylbu_4[i], lw=1.5,
                label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('Density')

    # D
    ax = axes[3]
    for i, pf in enumerate(POSITIONS):
        data = np.sort(all_errors[pf])
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf, color=rdylbu_4[i], lw=1.5,
                label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('CDF')
    ax.legend(loc='lower right')

    fig.tight_layout()
    path = os.path.join(OUT, f'tcm_rho{rho:.2f}_panels_signed.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}  (asym={asym:+.1f}, n={n_seeds})")


# ============================================================
# Figure 3: Supplementary — μ across rho
# ============================================================
print("\n\nSupplementary figures...")

model_mu_data = {}
model_se_data = {}
for rho in RHYLOS:
    mm = load_model_mu(LOGS_TCM, rho)
    model_mu_data[rho] = {
        'means': [np.mean(mm[pf]) for pf in POSITIONS],
        'sems': [np.std(mm[pf], ddof=1)/np.sqrt(max(len(mm[pf]),1))
                 for pf in POSITIONS],
        'n': len(mm[0.20]) if mm[0.20] else 0,
    }
    sm, _ = load_seed_means(LOGS_TCM, rho)
    model_se_data[rho] = {
        'means': [np.mean(sm[pf]) for pf in POSITIONS],
        'sems': [np.std(sm[pf], ddof=1)/np.sqrt(len(sm[pf]))
                 for pf in POSITIONS] if sm[0.20] else [0]*4,
        'n': len(sm[0.20]) if sm[0.20] else 0,
    }

rnn_mm = load_model_mu(LOGS_RNN, 0.0)
rnn_mu_means = [np.mean(rnn_mm[pf]) for pf in POSITIONS]
rnn_mu_sems = [np.std(rnn_mm[pf], ddof=1)/np.sqrt(max(len(rnn_mm[pf]),1))
               for pf in POSITIONS]

rnn_sm, _ = load_seed_means(LOGS_RNN, 0.0)
rnn_se_means_s = [np.mean(rnn_sm[pf]) for pf in POSITIONS]
rnn_se_sems_s = [np.std(rnn_sm[pf], ddof=1)/np.sqrt(len(rnn_sm[pf]))
                 for pf in POSITIONS]


def make_supp_fig(title, ylabel, human_means, human_sems, model_data,
                  rnn_m=None, rnn_s=None, outname='supp.png'):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Human
    ax.fill_between(PX,
                    [m - s for m, s in zip(human_means, human_sems)],
                    [m + s for m, s in zip(human_means, human_sems)],
                    color='black', alpha=0.10)
    ax.errorbar(PX, human_means, yerr=human_sems, fmt='s-',
                color='black', ms=8, lw=2, capsize=4, label='Human')

    for rho in RHYLOS:
        md = model_data[rho]
        color = rho_colors[rho]
        ax.fill_between(PX,
                        [m - s for m, s in zip(md['means'], md['sems'])],
                        [m + s for m, s in zip(md['means'], md['sems'])],
                        color=color, alpha=0.12)
        ax.errorbar(PX, md['means'], yerr=md['sems'], fmt='o-',
                    color=color, ms=6, lw=1.5, capsize=3,
                    label=f'TCM ρ={rho:.2f} (n={md["n"]})')

    if rnn_m is not None:
        ax.errorbar(PX, rnn_m, yerr=rnn_s if rnn_s else None,
                    fmt='D--', color='gray', ms=6, lw=1.5, capsize=3,
                    label='RNN')

    ax.axhline(0, color='black', linewidth=0.6, linestyle=':', alpha=0.4)
    ax.set_xlabel('Temporal Location', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(PX)
    ax.set_xticklabels(['20%', '40%', '60%', '80%'])
    ax.set_xlim(12, 88)
    ax.legend(fontsize=9, frameon=True, loc='lower left', ncol=2)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25, linestyle='--')

    fig.tight_layout()
    png_path = os.path.join(OUT, outname)
    pdf_path = os.path.join(OUT, outname.replace('.png', '.pdf'))
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {png_path}")


# Supp Fig A: μ
make_supp_fig(
    title='A: MemToolbox-estimated μ by temporal location',
    ylabel='μ',
    human_means=human_mu_means,
    human_sems=human_mu_sems,
    model_data=model_mu_data,
    rnn_m=rnn_mu_means,
    rnn_s=rnn_mu_sems if rnn_mu_sems else [0]*4,
    outname='supp_mu_across_rho.png',
)

# Supp Fig B: Signed error
make_supp_fig(
    title='B: Signed error by temporal location',
    ylabel='Signed Error (%)',
    human_means=human_se_means,
    human_sems=human_se_sems,
    model_data=model_se_data,
    rnn_m=rnn_se_means_s,
    rnn_s=rnn_se_sems_s if rnn_se_sems_s else [0]*4,
    outname='supp_signed_error_across_rho.png',
)

print("\nAll figures done.")
