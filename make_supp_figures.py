#!/usr/bin/env python3
"""Generate supplementary figures: μ across ρ values, signed error across ρ values."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from make_panels import (load_human_mu, load_human_signed_error,
                          load_model_mu, load_seed_means)

rhos = [0.60, 0.70, 0.80, 0.90, 0.95]
positions = [0.20, 0.40, 0.60, 0.80]
viridis = plt.cm.viridis
rho_colors = {r: viridis(i / (len(rhos) - 1)) for i, r in enumerate(rhos)}

px = np.array([20, 40, 60, 80])

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
})


def load_rnn_signed():
    """Load RNN baseline signed error by seed."""
    sm, _ = load_seed_means('logs/rnn', rho=0.0, seed_range=(42, 46))
    means = [np.mean(sm[pf]) for pf in positions]
    sems = [np.std(sm[pf], ddof=1) / np.sqrt(len(sm[pf])) for pf in positions]
    return means, sems


def load_rnn_mu():
    """Load RNN baseline μ by seed."""
    model_mu = load_model_mu('logs/rnn', rho=0.0, seed_range=(42, 46))
    means = [np.mean(model_mu[pf]) for pf in positions]
    sems = [np.std(model_mu[pf], ddof=1) / np.sqrt(max(len(model_mu[pf]), 1))
            for pf in positions]
    return means, sems


# ---- Load all data ----
human_mu = load_human_mu()
human_mu_means = [human_mu[pf]['mean'] for pf in positions]
human_mu_sems = [human_mu[pf]['sem'] for pf in positions]

human_se = load_human_signed_error()
human_se_means = [human_se[pf]['mean'] for pf in positions]
human_se_sems = [human_se[pf]['sem'] for pf in positions]

model_mu_data = {}
model_se_data = {}
for rho in rhos:
    sr = (42, 61) if rho == 0.95 else (42, 46)
    mm = load_model_mu('logs/global', rho=rho, seed_range=sr)
    model_mu_data[rho] = {
        'means': [np.mean(mm[pf]) for pf in positions],
        'sems': [np.std(mm[pf], ddof=1) / np.sqrt(max(len(mm[pf]), 1))
                 for pf in positions],
    }
    sm, _ = load_seed_means('logs/global', rho=rho, seed_range=sr)
    model_se_data[rho] = {
        'means': [np.mean(sm[pf]) for pf in positions],
        'sems': [np.std(sm[pf], ddof=1) / np.sqrt(len(sm[pf]))
                 for pf in positions],
    }

rnn_mu_means, rnn_mu_sems = load_rnn_mu()
rnn_se_means, rnn_se_sems = load_rnn_signed()


def make_figure(title, ylabel, human_means, human_sems, model_data,
                rnn_means=None, rnn_sems=None, outname='supp_fig.png'):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Human
    ax.fill_between(px,
                    [m - s for m, s in zip(human_means, human_sems)],
                    [m + s for m, s in zip(human_means, human_sems)],
                    color='black', alpha=0.10)
    ax.errorbar(px, human_means, yerr=human_sems, fmt='s-',
                color='black', ms=8, lw=2, capsize=4, label='Human')

    # Model per rho
    for rho in rhos:
        md = model_data[rho]
        color = rho_colors[rho]
        ax.fill_between(px,
                        [m - s for m, s in zip(md['means'], md['sems'])],
                        [m + s for m, s in zip(md['means'], md['sems'])],
                        color=color, alpha=0.12)
        ax.errorbar(px, md['means'], yerr=md['sems'], fmt='o-',
                    color=color, ms=6, lw=1.5, capsize=3,
                    label=f'TCM ρ={rho:.2f}')

    # RNN
    if rnn_means is not None:
        ax.errorbar(px, rnn_means, yerr=rnn_sems if rnn_sems else None,
                    fmt='D--', color='gray', ms=6, lw=1.5, capsize=3,
                    label='RNN')

    ax.axhline(0, color='black', linewidth=0.6, linestyle=':', alpha=0.4)
    ax.set_xlabel('Temporal Location', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(px)
    ax.set_xticklabels(['20%', '40%', '60%', '80%'])
    ax.set_xlim(12, 88)
    ax.legend(fontsize=9, frameon=True, loc='lower left', ncol=2)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25, linestyle='--')

    fig.tight_layout()
    fig.savefig(f'figures/{outname}', dpi=200, bbox_inches='tight')
    fig.savefig(f'figures/{outname.replace(".png", ".pdf")}',
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: figures/{outname}')


# ---- Figure 1: MemToolbox μ ----
make_figure(
    title='A: MemToolbox-estimated μ by temporal location',
    ylabel='μ',
    human_means=human_mu_means,
    human_sems=human_mu_sems,
    model_data=model_mu_data,
    rnn_means=rnn_mu_means,
    rnn_sems=rnn_mu_sems,
    outname='supp_mu_across_rho.png',
)

# ---- Figure 2: Signed error ----
make_figure(
    title='B: Signed error by temporal location',
    ylabel='Signed Error (%)',
    human_means=human_se_means,
    human_sems=human_se_sems,
    model_data=model_se_data,
    rnn_means=rnn_se_means,
    rnn_sems=rnn_se_sems,
    outname='supp_signed_error_across_rho.png',
)
