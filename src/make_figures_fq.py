#!/usr/bin/env python3
"""
Supplementary figure: f_q (item query) — MemToolbox μ and signed error.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import TEST_POSITIONS, HUMAN_ERRORS
from src.analysis import fit_memtoolbox_mu, load_model_mu, load_seed_means
from src.human_data import load_human_mu, load_human_signed_error

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white', 'font.size': 10,
})

POSITIONS = list(TEST_POSITIONS)
PX = [20, 40, 60, 80]
RHYLOS = [0.60, 0.70, 0.80, 0.90, 0.95]
viridis = plt.cm.viridis
RHO_COLORS = {r: viridis(i / (len(RHYLOS) - 1)) for i, r in enumerate(RHYLOS)}

LOGS_CQ = os.path.join(ROOT, 'logs', 'fixed_target')
LOGS_FQ = os.path.join(ROOT, 'logs', 'fq_query')
OUT = os.path.join(ROOT, 'figures')


# ---- Load all data ----
print("Loading c_q (context query) data...")
cq_mu = {}
cq_se = {}
for rho in RHYLOS:
    mm = load_model_mu(LOGS_CQ, rho)
    if mm[0.20]:
        cq_mu[rho] = {
            'means': [np.mean(mm[pf]) for pf in POSITIONS],
            'sems': [np.std(mm[pf], ddof=1)/np.sqrt(max(len(mm[pf]),1))
                     for pf in POSITIONS],
        }
    sm, _ = load_seed_means(LOGS_CQ, rho)
    if sm[0.20]:
        cq_se[rho] = {
            'means': [np.mean(sm[pf]) for pf in POSITIONS],
            'sems': [np.std(sm[pf], ddof=1)/np.sqrt(len(sm[pf]))
                     for pf in POSITIONS],
        }

print("Loading f_q (item query) data...")
fq_mu = {}
fq_se = {}
for rho in RHYLOS:
    mm = load_model_mu(LOGS_FQ, rho)
    if mm[0.20]:
        fq_mu[rho] = {
            'means': [np.mean(mm[pf]) for pf in POSITIONS],
            'sems': [np.std(mm[pf], ddof=1)/np.sqrt(max(len(mm[pf]),1))
                     for pf in POSITIONS],
        }
        print(f"  ρ={rho:.2f} μ: {[round(np.mean(mm[pf]),1) for pf in POSITIONS]}")
    else:
        print(f"  ρ={rho:.2f}: NO DATA")
    sm, _ = load_seed_means(LOGS_FQ, rho)
    if sm[0.20]:
        fq_se[rho] = {
            'means': [np.mean(sm[pf]) for pf in POSITIONS],
            'sems': [np.std(sm[pf], ddof=1)/np.sqrt(len(sm[pf]))
                     for pf in POSITIONS],
        }

human_mu = load_human_mu()
human_se = load_human_signed_error()
hm_means = [human_mu[pf]['mean'] for pf in POSITIONS]
hm_sems = [human_mu[pf]['sem'] for pf in POSITIONS]
hs_means = [human_se[pf]['mean'] for pf in POSITIONS]
hs_sems = [human_se[pf]['sem'] for pf in POSITIONS]


def make_fq_fig(title, ylabel, human_means, human_sems,
                 fq_data, outname):
    """f_q-only error curves with human overlaid."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Human
    if human_means:
        ax.fill_between(PX,
                        [m - s for m, s in zip(human_means, human_sems)],
                        [m + s for m, s in zip(human_means, human_sems)],
                        color='black', alpha=0.10)
        ax.errorbar(PX, human_means, yerr=human_sems, fmt='s-',
                    color='black', ms=8, lw=2, capsize=4, label='Human', zorder=6)

    # f_q only
    for rho in RHYLOS:
        if rho not in fq_data:
            continue
        d = fq_data[rho]
        color = RHO_COLORS[rho]
        ax.fill_between(PX,
                        [m - s for m, s in zip(d['means'], d['sems'])],
                        [m + s for m, s in zip(d['means'], d['sems'])],
                        color=color, alpha=0.12)
        ax.errorbar(PX, d['means'], yerr=d['sems'], fmt='o-',
                    color=color, ms=6, lw=2, capsize=3,
                    label=f'ρ={rho:.2f}')

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


print("\nGenerating f_q-only figures...")

make_fq_fig(
    title='Item-Query (f_q) — MemToolbox-estimated μ',
    ylabel='μ',
    human_means=hm_means,
    human_sems=hm_sems,
    fq_data=fq_mu,
    outname='supp_fq_mu.png',
)

make_fq_fig(
    title='Item-Query (f_q) — Signed Error',
    ylabel='Signed Error (%)',
    human_means=hs_means,
    human_sems=hs_sems,
    fq_data=fq_se,
    outname='supp_fq_signed.png',
)

print("\nDone.")
