"""
Endpoint-constrained TCM: per-interval-length comparison figure.
Shows signed error vs. true position for each interval length (8, 16, 32, 64)
as separate panels, with human data overlaid.

Usage:
  python make_viz_endpoints_intervals.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import load_results, discover_runs, parse_run_path
from src.config import TEST_POSITIONS, HUMAN_ERRORS, INTERVAL_LENGTHS_TEST

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.alpha': 0.15,
})

SIGMA = 0.05
RHOS = [0.95, 0.90, 0.80, 0.70]
INTERVALS = INTERVAL_LENGTHS_TEST
HUMAN_BLACK = '#222222'
LOGS_ROOT = os.path.join(os.path.dirname(__file__), 'logs', 'endpoints_v1_itemq')

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
_rho_norm = Normalize(vmin=0.65, vmax=1.0)
_spectral = plt.cm.Spectral_r
RHO_COLORS = [_spectral(_rho_norm(r)) for r in RHOS]


def load_interval_data(logs_root, rho, sigma, ilen):
    """Load errors for a specific interval length across seeds."""
    all_errs = {pf: [] for pf in TEST_POSITIONS}
    path_suffix = f'errors_i{ilen}.npz'
    for run_path in discover_runs(logs_root):
        info = parse_run_path(run_path)
        if (abs(info['rho'] - rho) > 0.005 or abs(info['sigma_m'] - sigma) > 0.005
            or info['d'] != 64):
            continue
        fpath = os.path.join(run_path, path_suffix)
        if not os.path.exists(fpath):
            continue
        data = np.load(fpath)
        for pf in TEST_POSITIONS:
            key = f'p{int(pf*100):02d}'
            if key in data:
                seed_mean = np.mean(data[key])
                all_errs[pf].append(seed_mean)

    if not all_errs[0.20]:
        return None, None, None, 0

    means = [np.mean(all_errs[pf]) for pf in TEST_POSITIONS]
    sds = [np.std(all_errs[pf]) for pf in TEST_POSITIONS]
    sems = [s / np.sqrt(len(all_errs[pf])) for s, _ in zip(sds, means)]
    n_seeds = len(all_errs[0.20])
    return means, sds, sems, n_seeds


# ---- Load data ----
print(f"Loading interval data from {LOGS_ROOT}...")
data = {}  # {ilen: {rho: {means, sds, sems, n_seeds}}}
for ilen in INTERVALS:
    data[ilen] = {}
    for rho in RHOS:
        means, sds, sems, n = load_interval_data(LOGS_ROOT, rho, SIGMA, ilen)
        if means is not None:
            asm = (abs(means[0])+abs(means[1])-abs(means[2])-abs(means[3]))/2
            print(f"  i{ilen} ρ={rho:.2f}: n={n} [{means[0]:.1f},{means[1]:.1f},{means[2]:.1f},{means[3]:.1f}] asm={asm:+.1f}")
            data[ilen][rho] = {'means': means, 'sds': sds, 'sems': sems, 'n_seeds': n}

# ---- Figure ----
n_intervals = len(INTERVALS)
fig, axes = plt.subplots(1, n_intervals, figsize=(5 * n_intervals, 5), squeeze=False)
x_pos = [20, 40, 60, 80]

for col, ilen in enumerate(INTERVALS):
    ax = axes[0, col]
    for i, rho in enumerate(RHOS):
        if rho not in data[ilen]:
            continue
        d = data[ilen][rho]
        means = d['means']; errs = d['sds']
        lower = [m - e for m, e in zip(means, errs)]
        upper = [m + e for m, e in zip(means, errs)]
        ax.fill_between(x_pos, lower, upper, color=RHO_COLORS[i], alpha=0.2, linewidth=0)
        ax.plot(x_pos, means, 'o-', color=RHO_COLORS[i],
                linewidth=2.0, markersize=6, alpha=0.9,
                markeredgecolor='white', markeredgewidth=0.5,
                label=f'ρ={rho} (n={d["n_seeds"]})')

    ax.plot(x_pos, HUMAN_ERRORS, 'D-', color=HUMAN_BLACK, linewidth=2.0, markersize=7,
            label='Human', zorder=5)
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.4)
    ax.set_xlabel('True Position (%)')
    if col == 0:
        ax.set_ylabel('Signed Error (%)')
    ax.set_title(f'Interval = {ilen}', fontsize=11, fontweight='bold')
    ax.set_xlim(10, 90)
    if col == n_intervals - 1:
        ax.legend(fontsize=6.5)

plt.suptitle(f'TCM Endpoint-Constrained — Per-Interval Error Patterns  |  σ_m = {SIGMA}',
             fontsize=13, fontweight='bold', y=1.01)

out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'tcm_viz_endpoints_intervals.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {out}")
