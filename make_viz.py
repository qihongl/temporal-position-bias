"""
Updated TCM visualization with multi-seed averaging.
Loads all seed results for a chosen sigma, computes means across seeds,
and plots with error bars. Nature-style color aesthetics.

Usage:
  python make_viz.py                          # default sigma=0.05
  python make_viz.py --sigma 0.10             # custom sigma
  python make_viz.py --sigma 0.10 --rhos 0.9,0.8,0.7
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

# ---- Nature-style aesthetics ----
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

# Position colours
POS_COLORS_PASTEL = ['#A6CEE3', '#B2DF8A', '#FB9A99', '#CAB2D6']
POS_COLORS_DARK   = ['#1F78B4', '#33A02C', '#E31A1C', '#6A3D9A']

# ---- Config ----
SIGMA = 0.05
RHOS = [0.95, 0.90, 0.80, 0.70]
HUMAN_BLACK = '#222222'
LOGS_ROOT = os.path.join(os.path.dirname(__file__), 'logs')


def load_seed_means(rho, sigma, d_filter=64):
    """Load errors for all seeds at given rho,sigma,d. Returns means, sds, sems, asm_sem."""
    all_errs = {pf: [] for pf in TEST_POSITIONS}
    all_asms = []
    for run_path in discover_runs(LOGS_ROOT):
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

# Nature-inspired spectral colours for rho continuum
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
_rho_norm = Normalize(vmin=0.65, vmax=1.0)
_spectral = plt.cm.Spectral_r
RHO_COLORS = [_spectral(_rho_norm(r)) for r in RHOS]


def load_seed_errors(rho, sigma, d_filter=64):
    """Load raw errors concatenated across all seeds for distributions."""
    combined = {pf: [] for pf in TEST_POSITIONS}
    for run_path in discover_runs(LOGS_ROOT):
        info = parse_run_path(run_path)
        if (abs(info['rho'] - rho) > 0.005 or abs(info['sigma_m'] - sigma) > 0.005
            or info['d'] != d_filter):
            continue
        errors, _, _ = load_results(run_path)
        for pf in TEST_POSITIONS:
            combined[pf].extend(errors[pf].tolist())
    for pf in TEST_POSITIONS:
        combined[pf] = np.array(combined[pf]) if combined[pf] else np.array([])
    return combined


def load_attn_averaged(rho, sigma):
    """Average raw attention weights (cosine similarity + softmax) across seeds."""
    import torch, torch.nn.functional as F

    n_seeds = 10
    all_w = {pf: [] for pf in TEST_POSITIONS}
    for seed in range(42, 52):
        torch.manual_seed(seed)
        items = torch.randn(1, SEQ_TEST, D)
        enc = TCMEncoder(rho, D, D)
        _, c_stack = enc(items)  # (1, L, D)
        c_norm = F.normalize(c_stack[0], dim=-1)
        for pf in TEST_POSITIONS:
            ti = int(pf * SEQ_TEST)
            cq = apply_noise(c_norm[ti:ti+1, :], sigma)
            scores = torch.mm(c_norm, cq.T).squeeze()
            w = F.softmax(scores, dim=0).detach().cpu().numpy()
            all_w[pf].append(w)

    # Average and SD
    avg_weights = {}
    sd_weights = {}
    for pf in TEST_POSITIONS:
        stacked = np.stack(all_w[pf], axis=0)
        avg_weights[pf] = stacked.mean(axis=0)
        sd_weights[pf] = stacked.std(axis=0)

    # Compute predicted position from averaged weights
    pos_tpl = np.linspace(0, 1, SEQ_TEST)
    result = {}
    for pf in TEST_POSITIONS:
        w = avg_weights[pf]
        pred = np.sum(w * pos_tpl) * 100
        result[pf] = {'weights': w, 'sd': sd_weights[pf], 'pred': pred, 'true_idx': int(pf * SEQ_TEST)}
    return result


def load_sim_averaged(rho, sigma):
    """Average context similarity matrix across seeds."""
    import torch, torch.nn.functional as F

    n_seeds = 10
    all_sims = []
    for seed in range(42, 52):
        torch.manual_seed(seed)
        items = torch.randn(1, SEQ_TEST, D)
        enc = TCMEncoder(rho, D, D)
        _, c_stack = enc(items)
        sim = (F.normalize(c_stack[0], dim=-1) @ F.normalize(c_stack[0], dim=-1).T).detach().cpu().numpy()
        all_sims.append(sim)
    stacked = np.stack(all_sims, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


# ============================================================
# Load data
# ============================================================
print(f"Loading multi-seed data for σ_m = {SIGMA}...")
data = {}
for rho in RHOS:
    means, sds, sems, asm_sem = load_seed_means(rho, SIGMA)
    errors = load_seed_errors(rho, SIGMA)
    if means is None:
        print(f"  ρ={rho:.2f}: NO DATA FOUND")
        continue
    n_seeds = len([r for r in discover_runs(LOGS_ROOT)
                    if abs(parse_run_path(r)['rho']-rho)<0.005
                    and abs(parse_run_path(r)['sigma_m']-SIGMA)<0.005
                    and parse_run_path(r)['d'] == 64])
    asm = (abs(means[0])+abs(means[1])-abs(means[2])-abs(means[3]))/2
    print(f"  ρ={rho:.2f}: {n_seeds} seeds, means=[{means[0]:.1f},{means[1]:.1f},{means[2]:.1f},{means[3]:.1f}], asm={asm:+.1f}")
    data[rho] = {'means': means, 'sds': sds, 'sems': sems, 'errors': errors, 'asm': asm, 'asm_sem': asm_sem, 'n_seeds': n_seeds}

# ============================================================
# FIGURE
# ============================================================
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# — A: Error curves with error bands —
ax = fig.add_subplot(gs[0, 0])
x_pos = [20, 40, 60, 80]
for i, rho in enumerate(RHOS):
    if rho not in data: continue
    d = data[rho]
    means = d['means']; errs = d['sds']  # use SD for visible band
    lower = [m - e for m, e in zip(means, errs)]
    upper = [m + e for m, e in zip(means, errs)]
    ax.fill_between(x_pos, lower, upper, color=RHO_COLORS[i], alpha=0.25, linewidth=0)
    ax.plot(x_pos, means, 'o-', color=RHO_COLORS[i],
            linewidth=2.2, markersize=7, alpha=0.95, markeredgecolor='white', markeredgewidth=0.5,
            label=f'\u03c1={rho} (n={d["n_seeds"]})')
ax.plot(x_pos, HUMAN_ERRORS, 'D-', color=HUMAN_BLACK, linewidth=2.2, markersize=8,
        label='Human', zorder=5)
ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.4)
ax.set_xlabel('True Position (%)'); ax.set_ylabel('Signed Error (%)')
ax.set_title(f'A  Error Pattern (\u03c3_m={SIGMA}, mean \u00b1 SD)', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7); ax.set_xlim(10, 90)

# — B: Asymmetry bar chart with SEM —
ax = fig.add_subplot(gs[0, 1])
rhos_present = [r for r in RHOS if r in data]
asm_vals = [data[r]['asm'] for r in rhos_present]
asm_sems = [data[r]['asm_sem'] for r in rhos_present]
bars = ax.bar(range(len(rhos_present)), asm_vals, color=[RHO_COLORS[RHOS.index(r)] for r in rhos_present],
              edgecolor='white', lw=1)
ax.errorbar(range(len(rhos_present)), asm_vals, yerr=asm_sems, fmt='none',
            color='black', capsize=5, linewidth=1.5, zorder=5)
ax.axhline(0, color='gray', ls='--', lw=0.8)
human_asm = (abs(HUMAN_ERRORS[0])+abs(HUMAN_ERRORS[1])-abs(HUMAN_ERRORS[2])-abs(HUMAN_ERRORS[3]))/2
ax.axhline(human_asm, color='black', ls=':', lw=1.5, label=f'Human: {human_asm:+.1f}')
ax.set_xticks(range(len(rhos_present)))
ax.set_xticklabels([f'\u03c1={r:.2f}\nn={data[r]["n_seeds"]}' for r in rhos_present])
ax.set_ylabel('Asymmetry (|early|\u2212|late|, %)')
ax.set_title('B  Asymmetry vs. \u03c1 (\u00b1SEM across seeds)', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)
for bar, v in zip(bars, asm_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.3 if v>=0 else v-0.8, f'{v:+.1f}', ha='center', fontsize=8, fontweight='bold')

# — C: Error distributions for best rho —
ax = fig.add_subplot(gs[0, 2])
rho_show = 0.95
if rho_show in data:
    for j, pf in enumerate(TEST_POSITIONS):
        es = data[rho_show]['errors'][pf]
        vp = ax.violinplot(es, positions=[j], vert=True, showmeans=True, showmedians=True, widths=0.7)
        for bd in vp['bodies']: bd.set_facecolor(POS_COLORS_PASTEL[j]); bd.set_alpha(0.6)
        for pn in ('cbars','cmins','cmaxes','cmeans','cmedians'):
            if pn in vp: vp[pn].set_color(POS_COLORS_DARK[j]); vp[pn].set_linewidth(0.8)
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(['20%','40%','60%','80%'])
ax.set_ylabel('Signed Error (%)')
ax.set_title(f'C  Error Distributions (\u03c1={rho_show}, pooled across seeds)', fontsize=11, fontweight='bold', loc='left')

# — D: Attention weights (averaged across seeds) —
ax = fig.add_subplot(gs[1, 0])
print(f"\nComputing attention weights averaged across seeds (ρ={rho_show})...")
attn_avg = load_attn_averaged(rho_show, SIGMA)
if attn_avg:
    x = np.arange(SEQ_TEST)
    for j, pf in enumerate(TEST_POSITIONS):
        if pf not in attn_avg: continue
        w = attn_avg[pf]['weights']; s = attn_avg[pf]['sd']
        ti = attn_avg[pf]['true_idx']; pr = attn_avg[pf]['pred']
        ax.fill_between(x, w - s, w + s, color=POS_COLORS_DARK[j], alpha=0.2, linewidth=0)
        ax.plot(x, w, color=POS_COLORS_DARK[j], lw=1.5, alpha=0.9, label=f"{int(pf*100)}% (pred={pr:.0f}%)")
        ax.axvline(ti, color=POS_COLORS_DARK[j], ls='--', lw=0.6, alpha=0.25)
ax.set_xlabel('Position index'); ax.set_ylabel('Attention weight')
ax.set_title(f'D  Attention Weights (\u03c1={rho_show}, avg. across seeds)', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)

# — E: Context similarity (averaged across seeds) —
ax = fig.add_subplot(gs[1, 1])
print(f"Computing similarity matrix averaged across seeds (ρ={rho_show})...")
sim, sim_sd = load_sim_averaged(rho_show, SIGMA)
im = ax.imshow(sim[:100, :100], aspect='equal', cmap='YlGnBu', vmin=0.5, vmax=1.0)
ax.set_xlabel('Context index j'); ax.set_ylabel('Context index i')
ax.set_title(f'E  cos(c_i, c_j)  (\u03c1={rho_show}, avg. across seeds)', fontsize=11, fontweight='bold', loc='left')
plt.colorbar(im, ax=ax, shrink=0.8); ax.axhline(50, color='black', ls='--', lw=1.5, alpha=0.7)

# — F: Forward vs backward (from averaged sim) —
ax = fig.add_subplot(gs[1, 2])
q = 50; ks = range(1, 21)
fw = np.array([sim[q, q+k] if q+k < 100 else np.nan for k in ks])
bw = np.array([sim[q, q-k] if q-k >= 0 else np.nan for k in ks])
fw_sd = np.array([sim_sd[q, q+k] if q+k < 100 else np.nan for k in ks])
bw_sd = np.array([sim_sd[q, q-k] if q-k >= 0 else np.nan for k in ks])

ax.fill_between(ks, fw - fw_sd, fw + fw_sd, color='#E6550D', alpha=0.2, linewidth=0)
ax.fill_between(ks, bw - bw_sd, bw + bw_sd, color='#3182BD', alpha=0.2, linewidth=0)
ax.plot(ks, fw, 'o-', color='#E6550D', lw=1.5, ms=4, label='Forward: cos(c_50, c_{50+k})')
ax.plot(ks, bw, 'o-', color='#3182BD', lw=1.5, ms=4, label='Backward: cos(c_50, c_{50-k})')
ax.axhline(1.0, color='gray', ls=':', lw=0.5)
ax.set_xlabel('|k| (lag)'); ax.set_ylabel('cosine similarity')
ax.set_ylim(0, 1)
ax.set_title(f'F  Forward vs. Backward Similarity (\u03c1={rho_show}, avg.)', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=8)

n_total = sum(d['n_seeds'] for d in data.values())
plt.suptitle(f'TCM Model — \u03c3_m = {SIGMA}  |  Mean across {n_total} seeds',
             fontsize=14, fontweight='bold', y=0.995)
out_dir = os.path.join(os.path.dirname(__file__), 'figures'); os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'tcm_viz_best_sigma.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {out}")
