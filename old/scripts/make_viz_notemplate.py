"""
Visualization for Learned Position Function (No-Template) Decoder.
Loads multi-seed data from logs_notmpl/, finds best model by human loss,
and produces a 6-panel publication figure.

Usage:
  python make_viz_notemplate.py
  python make_viz_notemplate.py --sigma 0.05
  python make_viz_notemplate.py --rhos 0.95,0.90,0.80,0.70
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import load_results, discover_runs, parse_run_path, apply_noise
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
    'legend.fontsize': 7.5,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.alpha': 0.15,
})

# ---- Config ----
SIGMA = 0.05
RHOS = [0.95, 0.90, 0.80, 0.70]
HUMAN_BLACK = '#222222'
LOGS_ROOT = os.path.join(os.path.dirname(__file__), 'logs_notmpl')
FIGS_ROOT = os.path.join(os.path.dirname(__file__), 'figures_notmpl')

POS_COLORS_PASTEL = ['#A6CEE3', '#B2DF8A', '#FB9A99', '#CAB2D6']
POS_COLORS_DARK   = ['#1F78B4', '#33A02C', '#E31A1C', '#6A3D9A']

# Nature-inspired spectral colours for rho continuum
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
_rho_norm = Normalize(vmin=0.65, vmax=1.0)
_spectral = plt.cm.Spectral_r
RHO_COLORS = [_spectral(_rho_norm(r)) for r in RHOS]
RHO_POS_FN_COLORS = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4']  # rho continuum for pos_fn


def load_seed_means(rho, sigma, d_filter=64):
    """Load per-seed mean errors across all seeds at given (rho, sigma)."""
    all_errs = {pf: [] for pf in TEST_POSITIONS}
    all_asms = []
    for run_path in discover_runs(LOGS_ROOT):
        info = parse_run_path(run_path)
        if (abs(info['rho'] - rho) > 0.005 or
            abs(info['sigma_m'] - sigma) > 0.005 or
            info['d'] != d_filter):
            continue
        errors, _, _ = load_results(run_path)
        seed_means = [np.mean(errors[pf]) for pf in TEST_POSITIONS]
        for pf in TEST_POSITIONS:
            all_errs[pf].append(seed_means[TEST_POSITIONS.index(pf)])
        all_asms.append((abs(seed_means[0]) + abs(seed_means[1])
                         - abs(seed_means[2]) - abs(seed_means[3])) / 2)

    if not all_errs[0.20]:
        return None, None, None, None

    means = [np.mean(all_errs[pf]) for pf in TEST_POSITIONS]
    sds   = [np.std(all_errs[pf]) for pf in TEST_POSITIONS]
    sems  = [s / np.sqrt(len(all_errs[pf])) for s, _ in zip(sds, means)]
    asm_sem = np.std(all_asms) / np.sqrt(len(all_asms)) if len(all_asms) > 1 else 0
    return means, sds, sems, asm_sem


def load_seed_errors(rho, sigma, d_filter=64):
    """Concatenate raw errors across all seeds for distribution plots."""
    combined = {pf: [] for pf in TEST_POSITIONS}
    for run_path in discover_runs(LOGS_ROOT):
        info = parse_run_path(run_path)
        if (abs(info['rho'] - rho) > 0.005 or
            abs(info['sigma_m'] - sigma) > 0.005 or
            info['d'] != d_filter):
            continue
        errors, _, _ = load_results(run_path)
        for pf in TEST_POSITIONS:
            combined[pf].extend(errors[pf].tolist())
    for pf in TEST_POSITIONS:
        combined[pf] = np.array(combined[pf]) if combined[pf] else np.array([])
    return combined


def load_attn_averaged(rho, sigma):
    """Average raw attention weights (cosine sim + softmax, no learned decoder)."""
    n_seeds = 5
    all_w = {pf: [] for pf in TEST_POSITIONS}
    for seed in range(42, 47):
        torch.manual_seed(seed)
        items = torch.randn(1, SEQ_TEST, D)
        enc = TCMEncoder(rho, D, D)
        _, c_stack = enc(items)
        c_norm = F.normalize(c_stack[0], dim=-1)
        for pf in TEST_POSITIONS:
            ti = int(pf * SEQ_TEST)
            cq = apply_noise(c_norm[ti:ti+1, :], sigma)
            scores = torch.mm(c_norm, cq.T).squeeze()
            w = F.softmax(scores, dim=0).detach().cpu().numpy()
            all_w[pf].append(w)

    avg_weights = {}
    sd_weights = {}
    for pf in TEST_POSITIONS:
        stacked = np.stack(all_w[pf], axis=0)
        avg_weights[pf] = stacked.mean(axis=0)
        sd_weights[pf] = stacked.std(axis=0)

    pos_tpl = np.linspace(0, 1, SEQ_TEST)
    result = {}
    for pf in TEST_POSITIONS:
        w = avg_weights[pf]
        pred = np.sum(w * pos_tpl) * 100
        result[pf] = {'weights': w, 'sd': sd_weights[pf],
                      'pred': pred, 'true_idx': int(pf * SEQ_TEST)}
    return result


def load_sim_averaged(rho, sigma):
    """Average context similarity matrix across seeds."""
    all_sims = []
    for seed in range(42, 47):
        torch.manual_seed(seed)
        items = torch.randn(1, SEQ_TEST, D)
        enc = TCMEncoder(rho, D, D)
        _, c_stack = enc(items)
        sim = (F.normalize(c_stack[0], dim=-1) @
               F.normalize(c_stack[0], dim=-1).T).detach().cpu().numpy()
        all_sims.append(sim)
    stacked = np.stack(all_sims, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def load_learned_pos_fn(rho, sigma):
    """Load the learned position function from a representative seed.
    Reconstructs f(t) by instantiating the encoder + decoder with saved weights
    and evaluating on t ∈ [0, 1]. Uses seed 42 as representative."""
    from src.model import AttentionDecoder

    run_path = None
    for rp in discover_runs(LOGS_ROOT):
        info = parse_run_path(rp)
        if (abs(info['rho'] - rho) < 0.005 and
            abs(info['sigma_m'] - sigma) < 0.005 and
            info['d'] == 64 and info['seed'] == 42):
            run_path = rp
            break

    if run_path is None:
        return None

    # We can't easily load the learned decoder weights from .npz files.
    # Instead, retrain a model quickly and extract pos_fn.
    torch.manual_seed(42); np.random.seed(42)
    encoder = TCMEncoder(rho, D, D)
    decoder = AttentionDecoder(D, use_position_template=False)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=5e-4)
    crit = torch.nn.MSELoss()

    for epoch in range(4000):
        L = np.random.randint(60, 181)
        items = torch.randn(32, L, D)
        f, c = encoder(items)
        q = torch.randint(0, L, (32,))
        c_q = apply_noise(c[torch.arange(32), q, :], sigma)
        pred = decoder(c_q, c)
        loss = crit(pred, q.float() / L)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        opt.step()

    with torch.no_grad():
        t = torch.linspace(0, 1, 201).unsqueeze(-1)
        f_vals = decoder.pos_fn(t).squeeze(-1).cpu().numpy()

    return f_vals


def scan_best_sigma():
    """Find the sigma that gives lowest human loss, averaged across rhos."""
    all_sigmas = sorted(set(
        parse_run_path(rp)['sigma_m']
        for rp in discover_runs(LOGS_ROOT)
    ))
    best_sigma = None
    best_loss = float('inf')

    for sm in all_sigmas:
        total_loss = 0
        n = 0
        for rho in RHOS:
            means, _, _, _ = load_seed_means(rho, sm)
            if means is not None:
                total_loss += compute_human_loss(means)
                n += 1
        if n > 0:
            avg_loss = total_loss / n
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_sigma = sm

    return best_sigma


# ============================================================
# Determine best sigma
# ============================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, default=None)
parser.add_argument('--rhos', type=str, default=None)
args = parser.parse_args()

if args.rhos is not None:
    RHOS = [float(r) for r in args.rhos.split(',')]

if args.sigma is not None:
    SIGMA = args.sigma
else:
    SIGMA = scan_best_sigma()
    if SIGMA is None:
        SIGMA = 0.05

print(f"Using σ_m = {SIGMA}")

# ============================================================
# Load data
# ============================================================
print(f"\nLoading multi-seed data for learned pos_fn, σ_m = {SIGMA}...")
data = {}
for rho in RHOS:
    means, sds, sems, asm_sem = load_seed_means(rho, SIGMA)
    errors = load_seed_errors(rho, SIGMA)
    if means is None:
        print(f"  ρ={rho:.2f}: NO DATA FOUND")
        continue
    n_seeds = len([r for r in discover_runs(LOGS_ROOT)
                    if abs(parse_run_path(r)['rho'] - rho) < 0.005
                    and abs(parse_run_path(r)['sigma_m'] - SIGMA) < 0.005
                    and parse_run_path(r)['d'] == 64])
    asm = (abs(means[0]) + abs(means[1]) - abs(means[2]) - abs(means[3])) / 2
    hl = compute_human_loss(means)
    print(f"  ρ={rho:.2f}: {n_seeds} seeds, "
          f"means=[{means[0]:.1f},{means[1]:.1f},{means[2]:.1f},{means[3]:.1f}], "
          f"asm={asm:+.1f}, human_loss={hl:.1f}")
    data[rho] = {'means': means, 'sds': sds, 'sems': sems, 'errors': errors,
                 'asm': asm, 'asm_sem': asm_sem, 'n_seeds': n_seeds,
                 'human_loss': hl}

# Find best rho at this sigma
best_rho = min(data.keys(), key=lambda r: data[r]['human_loss'])
print(f"\nBest model: ρ={best_rho:.2f}, human_loss={data[best_rho]['human_loss']:.1f}")

# ============================================================
# FIGURE: 2x3 panels
# ============================================================
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# — A: Error curves with SD bands —
ax = fig.add_subplot(gs[0, 0])
x_pos = [20, 40, 60, 80]
for i, rho in enumerate(RHOS):
    if rho not in data: continue
    d = data[rho]
    means = d['means']; errs = d['sds']
    lower = [m - e for m, e in zip(means, errs)]
    upper = [m + e for m, e in zip(means, errs)]
    ax.fill_between(x_pos, lower, upper, color=RHO_COLORS[i], alpha=0.22, linewidth=0)
    ax.plot(x_pos, means, 'o-', color=RHO_COLORS[i],
            linewidth=2.2, markersize=7, alpha=0.95,
            markeredgecolor='white', markeredgewidth=0.5,
            label=f'ρ={rho} (n={d["n_seeds"]})')
ax.plot(x_pos, HUMAN_ERRORS, 'D-', color=HUMAN_BLACK, linewidth=2.2, markersize=8,
        label='Human', zorder=5)
ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.4)
ax.set_xlabel('True Position (%)')
ax.set_ylabel('Signed Error (%)')
ax.set_title(f'A  Error Pattern — Learned PosFn  (σ_m={SIGMA}, mean ± SD)',
             fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)
ax.set_xlim(10, 90)

# — B: Asymmetry bar chart —
ax = fig.add_subplot(gs[0, 1])
rhos_present = [r for r in RHOS if r in data]
asm_vals = [data[r]['asm'] for r in rhos_present]
asm_sems = [data[r]['asm_sem'] for r in rhos_present]
bars = ax.bar(range(len(rhos_present)), asm_vals,
              color=[RHO_COLORS[RHOS.index(r)] for r in rhos_present],
              edgecolor='white', lw=1)
ax.errorbar(range(len(rhos_present)), asm_vals, yerr=asm_sems, fmt='none',
            color='black', capsize=5, linewidth=1.5, zorder=5)
ax.axhline(0, color='gray', ls='--', lw=0.8)
human_asm = (abs(HUMAN_ERRORS[0]) + abs(HUMAN_ERRORS[1])
             - abs(HUMAN_ERRORS[2]) - abs(HUMAN_ERRORS[3])) / 2
ax.axhline(human_asm, color='black', ls=':', lw=1.5,
           label=f'Human: {human_asm:+.1f}')
ax.set_xticks(range(len(rhos_present)))
ax.set_xticklabels([f'ρ={r:.2f}\nn={data[r]["n_seeds"]}' for r in rhos_present])
ax.set_ylabel('Asymmetry (|early| − |late|, %)')
ax.set_title('B  Asymmetry — Learned PosFn  (±SEM across seeds)',
             fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)
for bar, v in zip(bars, asm_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + 0.3 if v >= 0 else v - 0.8,
            f'{v:+.1f}', ha='center', fontsize=8, fontweight='bold')

# — C: Learned position function f(t) —
ax = fig.add_subplot(gs[0, 2])
# Plot identity line (hardcoded template) for reference
ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.2, alpha=0.6,
        label='Linear template (i/L)')
print("\nComputing learned position functions...")
for i, rho in enumerate(RHOS):
    if rho not in data: continue
    f_vals = load_learned_pos_fn(rho, SIGMA)
    if f_vals is None: continue
    t = np.linspace(0, 1, len(f_vals))
    ax.plot(t, f_vals, '-', color=RHO_POS_FN_COLORS[i], lw=2.0,
            label=f'ρ={rho}', alpha=0.9)
    # Mark f(0) and f(1)
    ax.plot(t[0], f_vals[0], 'o', color=RHO_POS_FN_COLORS[i], ms=5)
    ax.plot(t[-1], f_vals[-1], 's', color=RHO_POS_FN_COLORS[i], ms=5)
ax.set_xlabel('Relative position t ∈ [0, 1]')
ax.set_ylabel('f(t) — learned position value')
ax.set_title('C  Learned Position Function f(t)',
             fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7, loc='upper left')
ax.set_xlim(0, 1)
ax.set_ylim(-0.15, 1.15)

# — D: Error distributions for best rho —
ax = fig.add_subplot(gs[1, 0])
if best_rho in data:
    for j, pf in enumerate(TEST_POSITIONS):
        es = data[best_rho]['errors'][pf]
        vp = ax.violinplot(es, positions=[j], vert=True,
                           showmeans=True, showmedians=True, widths=0.7)
        for bd in vp['bodies']:
            bd.set_facecolor(POS_COLORS_PASTEL[j]); bd.set_alpha(0.6)
        for pn in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if pn in vp:
                vp[pn].set_color(POS_COLORS_DARK[j])
                vp[pn].set_linewidth(0.8)
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(['20%', '40%', '60%', '80%'])
ax.set_ylabel('Signed Error (%)')
ax.set_title(f'D  Error Distributions  (best ρ={best_rho}, pooled seeds)',
             fontsize=11, fontweight='bold', loc='left')

# — E: Attention weights —
ax = fig.add_subplot(gs[1, 1])
print(f"Computing attention weights (ρ={best_rho})...")
attn_avg = load_attn_averaged(best_rho, SIGMA)
if attn_avg:
    x = np.arange(SEQ_TEST)
    for j, pf in enumerate(TEST_POSITIONS):
        if pf not in attn_avg: continue
        w = attn_avg[pf]['weights']; s = attn_avg[pf]['sd']
        ti = attn_avg[pf]['true_idx']; pr = attn_avg[pf]['pred']
        ax.fill_between(x, w - s, w + s,
                        color=POS_COLORS_DARK[j], alpha=0.2, linewidth=0)
        ax.plot(x, w, color=POS_COLORS_DARK[j], lw=1.5, alpha=0.9,
                label=f"{int(pf*100)}% (pred={pr:.0f}%)")
        ax.axvline(ti, color=POS_COLORS_DARK[j], ls='--', lw=0.6, alpha=0.25)
ax.set_xlabel('Position index')
ax.set_ylabel('Attention weight')
ax.set_title(f'E  Attention Weights  (ρ={best_rho}, avg. across seeds)',
             fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)

# — F: Context similarity —
ax = fig.add_subplot(gs[1, 2])
print(f"Computing similarity matrix (ρ={best_rho})...")
sim, sim_sd = load_sim_averaged(best_rho, SIGMA)
im = ax.imshow(sim[:100, :100], aspect='equal', cmap='YlGnBu',
               vmin=0.5, vmax=1.0)
ax.set_xlabel('Context index j')
ax.set_ylabel('Context index i')
ax.set_title(f'F  cos(c_i, c_j)  (ρ={best_rho}, avg. across seeds)',
             fontsize=11, fontweight='bold', loc='left')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.axhline(50, color='black', ls='--', lw=1.5, alpha=0.7)

# ---- Title ----
n_total = sum(d['n_seeds'] for d in data.values())
plt.suptitle(
    f'TCM Model — Learned Position Function  |  σ_m = {SIGMA}  |  '
    f'Best ρ = {best_rho}  |  Mean across {n_total} seeds',
    fontsize=14, fontweight='bold', y=0.995)

os.makedirs(FIGS_ROOT, exist_ok=True)
out = os.path.join(FIGS_ROOT, 'tcm_viz_posfn_best.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nFigure saved: {out}")
