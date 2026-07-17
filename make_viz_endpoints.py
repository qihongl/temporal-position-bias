"""
Endpoint-Constrained TCM visualization with multi-seed averaging.
Compares the original (global attention) model with the endpoint-constrained variant.

Usage:
  python make_viz_endpoints.py
  python make_viz_endpoints.py --sigma 0.05
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import load_results, discover_runs, parse_run_path, apply_noise, ensure_dir
from src.config import TEST_POSITIONS, HUMAN_ERRORS, SEQ_TEST, D_ITEM as D
from src.model import TCMEncoder
from src.analysis import compute_human_loss

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.alpha': 0.15,
})

SIGMA = 0.05
RHOS = [0.95, 0.90, 0.80, 0.70]
HUMAN_BLACK = '#222222'

LOGS_ORIGINAL = os.path.join(os.path.dirname(__file__), 'logs', 'global')
LOGS_ENDPOINTS = os.path.join(os.path.dirname(__file__), 'logs', 'endpoints_v1_itemq')

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
_rho_norm = Normalize(vmin=0.65, vmax=1.0)
_spectral = plt.cm.Spectral_r
RHO_COLORS = [_spectral(_rho_norm(r)) for r in RHOS]


def load_seed_means(logs_root, rho, sigma, d_filter=64):
    all_errs = {pf: [] for pf in TEST_POSITIONS}
    all_asms = []
    for run_path in discover_runs(logs_root):
        info = parse_run_path(run_path)
        if (abs(info['rho'] - rho) > 0.005 or abs(info['sigma_m'] - sigma) > 0.005
            or info['d'] != d_filter):
            continue
        errors, _, _ = load_results(run_path)
        seed_means = [np.mean(errors[pf]) for pf in TEST_POSITIONS]
        for pf in TEST_POSITIONS:
            all_errs[pf].append(seed_means[TEST_POSITIONS.index(pf)])
        all_asms.append((abs(seed_means[0])+abs(seed_means[1])-abs(seed_means[2])-abs(seed_means[3]))/2)

    if not all_errs[0.20]:
        return None, None, None, None

    means = [np.mean(all_errs[pf]) for pf in TEST_POSITIONS]
    sds = [np.std(all_errs[pf]) for pf in TEST_POSITIONS]
    sems = [s / np.sqrt(len(all_errs[pf])) for s, _ in zip(sds, means)]
    asm_sem = np.std(all_asms) / np.sqrt(len(all_asms)) if len(all_asms) > 1 else 0
    return means, sds, sems, asm_sem


def load_data(logs_root):
    data = {}
    for rho in RHOS:
        means, sds, sems, asm_sem = load_seed_means(logs_root, rho, SIGMA)
        if means is None:
            print(f"  ρ={rho:.2f}: NO DATA")
            continue
        n_seeds = len([r for r in discover_runs(logs_root)
                        if abs(parse_run_path(r)['rho']-rho)<0.005
                        and abs(parse_run_path(r)['sigma_m']-SIGMA)<0.005
                        and parse_run_path(r)['d'] == 64])
        asm = (abs(means[0])+abs(means[1])-abs(means[2])-abs(means[3]))/2
        print(f"  ρ={rho:.2f}: {n_seeds} seeds, means=[{means[0]:.1f},{means[1]:.1f},{means[2]:.1f},{means[3]:.1f}], asm={asm:+.1f}")
        data[rho] = {'means': means, 'sds': sds, 'sems': sems, 'asm': asm, 'asm_sem': asm_sem, 'n_seeds': n_seeds}
    return data


# ---- Load data ----
print("Original model (global attention):")
data_orig = load_data(LOGS_ORIGINAL)
print("\nEndpoint-constrained model:")
data_ep = load_data(LOGS_ENDPOINTS)

# ---- Figure ----
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# — A: Error curves — original and endpoint overlaid —
ax = fig.add_subplot(gs[0, 0])
x_pos = [20, 40, 60, 80]

for i, rho in enumerate(RHOS):
    if rho not in data_ep:
        continue
    d = data_ep[rho]
    means = d['means']; errs = d['sds']
    lower = [m - e for m, e in zip(means, errs)]
    upper = [m + e for m, e in zip(means, errs)]
    ax.fill_between(x_pos, lower, upper, color=RHO_COLORS[i], alpha=0.2, linewidth=0)
    ax.plot(x_pos, means, 'o-', color=RHO_COLORS[i],
            linewidth=2.2, markersize=7, alpha=0.95, markeredgecolor='white', markeredgewidth=0.5,
            label=f'ρ={rho} EP (n={d["n_seeds"]})')

# Original as dashed
for i, rho in enumerate(RHOS):
    if rho in data_orig:
        d = data_orig[rho]
        ax.plot(x_pos, d['means'], 's--', color=RHO_COLORS[i],
                linewidth=1.2, markersize=6, alpha=0.5,
                label=f'ρ={rho} Orig (n={d["n_seeds"]})')

ax.plot(x_pos, HUMAN_ERRORS, 'D-', color=HUMAN_BLACK, linewidth=2.2, markersize=8,
        label='Human', zorder=5)
ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.4)
ax.set_xlabel('True Position (%)'); ax.set_ylabel('Signed Error (%)')
ax.set_title('A  Error Pattern — Endpoint vs. Original (mean ± SD)', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=6.5); ax.set_xlim(10, 90)

# — B: Asymmetry comparison —
ax = fig.add_subplot(gs[0, 1])
rhos_present = [r for r in RHOS if r in data_ep and r in data_orig]
x = np.arange(len(rhos_present))
w = 0.35

asm_ep = [data_ep[r]['asm'] for r in rhos_present]
asm_sem_ep = [data_ep[r]['asm_sem'] for r in rhos_present]
asm_orig = [data_orig[r]['asm'] for r in rhos_present]
asm_sem_orig = [data_orig[r]['asm_sem'] for r in rhos_present]

bars_ep = ax.bar(x - w/2, asm_ep, w, color=[RHO_COLORS[RHOS.index(r)] for r in rhos_present],
                 edgecolor='white', lw=1, label='Endpoint')
bars_orig = ax.bar(x + w/2, asm_orig, w, color=[RHO_COLORS[RHOS.index(r)] for r in rhos_present],
                   edgecolor='white', lw=1, alpha=0.4, label='Original')

ax.errorbar(x - w/2, asm_ep, yerr=asm_sem_ep, fmt='none', color='black', capsize=5, linewidth=1.5, zorder=5)
ax.errorbar(x + w/2, asm_orig, yerr=asm_sem_orig, fmt='none', color='gray', capsize=5, linewidth=1.5, zorder=5)

ax.axhline(0, color='gray', ls='--', lw=0.8)
human_asm = (abs(HUMAN_ERRORS[0])+abs(HUMAN_ERRORS[1])-abs(HUMAN_ERRORS[2])-abs(HUMAN_ERRORS[3]))/2
ax.axhline(human_asm, color='black', ls=':', lw=1.5, label=f'Human: {human_asm:+.1f}')
ax.set_xticks(x)
ax.set_xticklabels([f'ρ={r:.2f}' for r in rhos_present])
ax.set_ylabel('Asymmetry (|early|−|late|, %)')
ax.set_title('B  Asymmetry — Endpoint vs. Original (±SEM)', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)

# — C: Error distributions for best rho (endpoint model) —
ax = fig.add_subplot(gs[0, 2])
rho_show = 0.95
if rho_show in data_ep:
    errors_ep = {}
    for run_path in discover_runs(LOGS_ENDPOINTS):
        info = parse_run_path(run_path)
        if (abs(info['rho'] - rho_show) > 0.005 or abs(info['sigma_m'] - SIGMA) > 0.005
            or info['d'] != 64):
            continue
        errs, _, _ = load_results(run_path)
        for pf in TEST_POSITIONS:
            errors_ep.setdefault(pf, []).extend(errs[pf].tolist())

    from matplotlib.cm import ScalarMappable
    pos_colors_pastel = ['#A6CEE3', '#B2DF8A', '#FB9A99', '#CAB2D6']
    pos_colors_dark = ['#1F78B4', '#33A02C', '#E31A1C', '#6A3D9A']
    for j, pf in enumerate(TEST_POSITIONS):
        if pf in errors_ep:
            es = np.array(errors_ep[pf])
            vp = ax.violinplot(es, positions=[j], vert=True, showmeans=True, showmedians=True, widths=0.7)
            for bd in vp['bodies']: bd.set_facecolor(pos_colors_pastel[j]); bd.set_alpha(0.6)
            for pn in ('cbars','cmins','cmaxes','cmeans','cmedians'):
                if pn in vp: vp[pn].set_color(pos_colors_dark[j]); vp[pn].set_linewidth(0.8)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(['20%','40%','60%','80%'])
    ax.set_ylabel('Signed Error (%)')
    ax.set_title(f'C  Error Distributions (Endpoint ρ={rho_show})', fontsize=11, fontweight='bold', loc='left')

# — D: Endpoint vs Original error comparison per position —
ax = fig.add_subplot(gs[1, 0])
if rho_show in data_ep and rho_show in data_orig:
    ep_means = data_ep[rho_show]['means']
    ep_sems = data_ep[rho_show]['sems']
    orig_means = data_orig[rho_show]['means']
    orig_sems = data_orig[rho_show]['sems']

    x_p = np.arange(4)
    w_p = 0.35
    ax.bar(x_p - w_p/2, ep_means, w_p, color='#E6550D', edgecolor='white', lw=1, label='Endpoint')
    ax.bar(x_p + w_p/2, orig_means, w_p, color='#3182BD', edgecolor='white', lw=1, label='Original')
    ax.errorbar(x_p - w_p/2, ep_means, yerr=ep_sems, fmt='none', color='black', capsize=5, linewidth=1.5)
    ax.errorbar(x_p + w_p/2, orig_means, yerr=orig_sems, fmt='none', color='black', capsize=5, linewidth=1.5)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xticks(x_p); ax.set_xticklabels(['20%','40%','60%','80%'])
    ax.set_ylabel('Signed Error (%)')
    ax.set_title(f'D  Per-Position Comparison (ρ={rho_show})', fontsize=11, fontweight='bold', loc='left')
    ax.legend(fontsize=8)

# — E: Asymmetry vs rho — endpoint only —
ax = fig.add_subplot(gs[1, 1])
rhos_ep = [r for r in RHOS if r in data_ep]
asm_vals_ep = [data_ep[r]['asm'] for r in rhos_ep]
asm_sems_ep = [data_ep[r]['asm_sem'] for r in rhos_ep]
bars = ax.bar(range(len(rhos_ep)), asm_vals_ep, color=[RHO_COLORS[RHOS.index(r)] for r in rhos_ep],
              edgecolor='white', lw=1)
ax.errorbar(range(len(rhos_ep)), asm_vals_ep, yerr=asm_sems_ep, fmt='none',
            color='black', capsize=5, linewidth=1.5, zorder=5)
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axhline(human_asm, color='black', ls=':', lw=1.5, label=f'Human: {human_asm:+.1f}')
ax.set_xticks(range(len(rhos_ep)))
ax.set_xticklabels([f'ρ={r:.2f}\nn={data_ep[r]["n_seeds"]}' for r in rhos_ep])
ax.set_ylabel('Asymmetry (|early|−|late|, %)')
ax.set_title('E  Endpoint Asymmetry vs. ρ (±SEM)', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)
for bar, v in zip(bars, asm_vals_ep):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.3 if v>=0 else v-0.8, f'{v:+.1f}', ha='center', fontsize=8, fontweight='bold')

# — F: Human loss comparison —
ax = fig.add_subplot(gs[1, 2])
rhos_all = [r for r in RHOS if r in data_ep or r in data_orig]
loss_ep = [compute_human_loss(data_ep[r]['means']) if r in data_ep else np.nan for r in rhos_all]
loss_orig = [compute_human_loss(data_orig[r]['means']) if r in data_orig else np.nan for r in rhos_all]

x_l = np.arange(len(rhos_all))
w_l = 0.35
bars_ep_l = ax.bar(x_l - w_l/2, loss_ep, w_l, color='#E6550D', edgecolor='white', lw=1, label='Endpoint')
bars_orig_l = ax.bar(x_l + w_l/2, loss_orig, w_l, color='#3182BD', edgecolor='white', lw=1, label='Original')
ax.set_xticks(x_l)
ax.set_xticklabels([f'ρ={r:.2f}' for r in rhos_all])
ax.set_ylabel('Human MSE Loss')
ax.set_title('F  Fit to Human Data (lower = better)', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=8)

n_ep = sum(d['n_seeds'] for d in data_ep.values())
n_orig = sum(d['n_seeds'] for d in data_orig.values())
plt.suptitle(f'TCM Model — Endpoint-Constrained vs. Original  |  σ_m = {SIGMA}  |  EP: {n_ep} seeds  Orig: {n_orig} seeds',
             fontsize=14, fontweight='bold', y=0.995)
out_dir = os.path.join(os.path.dirname(__file__), 'figures'); os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'tcm_viz_endpoints.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {out}")
