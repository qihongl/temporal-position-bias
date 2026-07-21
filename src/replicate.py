#!/usr/bin/env python3
"""
Full replication: train all models (5 seeds) and regenerate every figure
from figures/ into figures_replication/.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import gc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

from src.train import train_model, evaluate_model, train_rnn_model
from src.analysis import compute_summary_stats, compute_human_loss, \
    fit_memtoolbox_mu, load_model_mu, load_seed_means
from src.human_data import load_human_mu, load_human_signed_error
from src.utils import log_dir, save_results
from src.config import D_ITEM, D_CONTEXT, TEST_POSITIONS, HUMAN_ERRORS

sns.set_context('talk')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
})

# ── Settings ──────────────────────────────────────────────
RHYLOS = [0.60, 0.70, 0.80, 0.90, 0.95]
SIGMA_M = 0.05
N_SEEDS = 5
D = D_CONTEXT
LOGS = os.path.join(ROOT, 'logs', 'replication')
OUT = os.path.join(ROOT, 'figures_replication')
os.makedirs(OUT, exist_ok=True)

POSITIONS = list(TEST_POSITIONS)
PX = [20, 40, 60, 80]
rdylbu_4 = ['#D73027', '#F46D43', '#74ADD1', '#4575B4']
viridis_rho = {r: plt.cm.viridis(i / (len(RHYLOS)-1)) for i, r in enumerate(RHYLOS)}

# ── Train ─────────────────────────────────────────────────
total = len(RHYLOS) * N_SEEDS + N_SEEDS  # TCM + RNN
print("=" * 55)
print(f"TRAINING: {len(RHYLOS)} rhos × {N_SEEDS} seeds (TCM) + {N_SEEDS} seeds (RNN) = {total} models")
print(f"  σ_m = {SIGMA_M}, logs → {LOGS}")
print("=" * 55)

counter = 0
for rho in RHYLOS:
    for seed in range(42, 42 + N_SEEDS):
        counter += 1
        ldir = log_dir(rho, SIGMA_M, D, seed, base=LOGS)
        if os.path.exists(os.path.join(ldir, 'errors.npz')):
            print(f"[{counter}/{total}] TCM ρ={rho:.2f} seed={seed} — skip", flush=True)
            continue

        encoder, decoder, train_mse = train_model(
            rho, SIGMA_M, seed=seed, d_item=D_ITEM, d_context=D)
        errors = evaluate_model(encoder, decoder, SIGMA_M, d_item=D_ITEM)
        means, _, asymmetry = compute_summary_stats(errors)
        loss_val = compute_human_loss(means)

        save_results(ldir, errors, {}, rho, SIGMA_M,
                     metadata={'epochs': 4000, 'loss': loss_val,
                               'asymmetry': asymmetry, 'd': D, 'seed': seed,
                               'train_mse': train_mse})
        print(f"[{counter}/{total}] TCM ρ={rho:.2f} seed={seed} — "
              f"[{means[0]:.1f} {means[1]:.1f} {means[2]:.1f} {means[3]:.1f}] asym={asymmetry:+.1f}",
              flush=True)
        del encoder, decoder; gc.collect()

# RNN control
RNN_RHO = 0.0
for seed in range(42, 42 + N_SEEDS):
    counter += 1
    ldir = log_dir(RNN_RHO, SIGMA_M, D, seed, base=LOGS)
    if os.path.exists(os.path.join(ldir, 'errors.npz')):
        print(f"[{counter}/{total}] RNN seed={seed} — skip", flush=True)
        continue

    encoder, decoder, train_mse = train_rnn_model(
        SIGMA_M, seed=seed, d_item=D_ITEM, d_context=D)
    errors = evaluate_model(encoder, decoder, SIGMA_M, d_item=D_ITEM)
    means, _, asymmetry = compute_summary_stats(errors)

    save_results(ldir, errors, {}, RNN_RHO, SIGMA_M,
                 metadata={'epochs': 4000, 'loss': 0, 'asymmetry': asymmetry,
                           'd': D, 'seed': seed, 'model_type': 'rnn',
                           'train_mse': train_mse})
    print(f"[{counter}/{total}] RNN seed={seed} — "
          f"[{means[0]:.1f} {means[1]:.1f} {means[2]:.1f} {means[3]:.1f}] asym={asymmetry:+.1f}",
          flush=True)
    del encoder, decoder; gc.collect()

print("\nTraining complete.\n")

# ── Load human data ───────────────────────────────────────
human_mu = load_human_mu()
human_se = load_human_signed_error()
hm = [human_mu[pf]['mean'] for pf in POSITIONS]
hm_s = [human_mu[pf]['sem'] for pf in POSITIONS]
hs = [human_se[pf]['mean'] for pf in POSITIONS]
hs_s = [human_se[pf]['sem'] for pf in POSITIONS]

# ── Load RNN data ─────────────────────────────────────────
rnn_model_mu = load_model_mu(LOGS, 0.0)
rnn_mu_means = [np.mean(rnn_model_mu[pf]) for pf in POSITIONS] if rnn_model_mu[0.20] else None
rnn_mu_sems = [np.std(rnn_model_mu[pf], ddof=1) / np.sqrt(max(len(rnn_model_mu[pf]), 1))
               for pf in POSITIONS] if rnn_model_mu[0.20] else None

rnn_se, _ = load_seed_means(LOGS, 0.0)
rnn_se_means = [np.mean(rnn_se[pf]) for pf in POSITIONS] if rnn_se[0.20] else None
rnn_se_sems = [np.std(rnn_se[pf], ddof=1) / np.sqrt(len(rnn_se[pf]))
               for pf in POSITIONS] if rnn_se[0.20] else None

# ── Panels: mu mode ───────────────────────────────────────
print("Generating panels (mu mode)...")
for rho in RHYLOS:
    seed_means, all_errors = load_seed_means(LOGS, rho)
    model_mu = load_model_mu(LOGS, rho)
    n = len(seed_means[0.20])
    if n == 0:
        print(f"  ρ={rho:.2f}: NO DATA"); continue

    mu_means = [np.mean(model_mu[pf]) for pf in POSITIONS]
    mu_sems = [np.std(model_mu[pf], ddof=1)/np.sqrt(max(len(model_mu[pf]),1))
               for pf in POSITIONS]
    asym = (abs(mu_means[0])+abs(mu_means[1])-abs(mu_means[2])-abs(mu_means[3]))/2

    seed_early = [(abs(model_mu[0.20][i])+abs(model_mu[0.40][i]))/2 for i in range(n)]
    seed_late  = [(abs(model_mu[0.60][i])+abs(model_mu[0.80][i]))/2 for i in range(n)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    ax = axes[0]
    ax.fill_between(PX, [m-s for m,s in zip(mu_means,mu_sems)],
                    [m+s for m,s in zip(mu_means,mu_sems)], color='#E6550D', alpha=0.15)
    ax.errorbar(PX, mu_means, yerr=mu_sems, fmt='o-', color='#E6550D', lw=1.5, ms=5, capsize=3,
                label=f'TCM (n={n})')
    if rnn_mu_means is not None:
        ax.errorbar(PX, rnn_mu_means, yerr=rnn_mu_sems, fmt='o-', color='gray',
                    lw=1.5, ms=5, capsize=3, label='RNN')
    ax.fill_between(PX, [m-s for m,s in zip(hm,hm_s)],
                    [m+s for m,s in zip(hm,hm_s)], color='black', alpha=0.10)
    ax.errorbar(PX, hm, yerr=hm_s, fmt='s-', color='black', ms=5, lw=1.5, capsize=3,
                label='Human (μ)')
    ax.set_xlabel('True Position (%)'); ax.set_ylabel('μ')
    ax.legend(fontsize=8); ax.set_xticks(PX)

    ax = axes[1]
    ax.bar([0,1], [np.mean(seed_early), np.mean(seed_late)],
           color=['#D73027','#4575B4'], edgecolor='white', lw=1, width=0.5)
    ax.errorbar([0,1], [np.mean(seed_early), np.mean(seed_late)],
                yerr=[np.std(seed_early,ddof=1)/np.sqrt(n),
                      np.std(seed_late,ddof=1)/np.sqrt(n)],
                fmt='none', color='black', capsize=5, lw=1)
    ax.set_xticks([0,1]); ax.set_xticklabels(['20%+40%','60%+80%'])

    ax = axes[2]
    x_all = np.concatenate([all_errors[pf] for pf in POSITIONS])
    xs = np.linspace(x_all.min()-5, x_all.max()+5, 200)
    for i, pf in enumerate(POSITIONS):
        kde = gaussian_kde(np.array(all_errors[pf]))
        ax.fill_between(xs, kde(xs), alpha=0.3, color=rdylbu_4[i])
        ax.plot(xs, kde(xs), color=rdylbu_4[i], lw=1.5, label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('Density')

    ax = axes[3]
    for i, pf in enumerate(POSITIONS):
        data = np.sort(all_errors[pf])
        ax.plot(data, np.arange(1,len(data)+1)/len(data),
                color=rdylbu_4[i], lw=1.5, label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('CDF')
    ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'tcm_rho{rho:.2f}_panels_mu.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  tcm_rho{rho:.2f}_panels_mu.png  (asym={asym:+.1f}, n={n})")

# ── Panels: signed mode ───────────────────────────────────
print("\nGenerating panels (signed mode)...")
for rho in RHYLOS:
    seed_means, all_errors = load_seed_means(LOGS, rho)
    n = len(seed_means[0.20])
    if n == 0: continue

    se_means = [np.mean(seed_means[pf]) for pf in POSITIONS]
    se_sems = [np.std(seed_means[pf],ddof=1)/np.sqrt(len(seed_means[pf]))
               for pf in POSITIONS]
    asym = (abs(se_means[0])+abs(se_means[1])-abs(se_means[2])-abs(se_means[3]))/2

    seed_early = [(abs(seed_means[0.20][i])+abs(seed_means[0.40][i]))/2 for i in range(n)]
    seed_late  = [(abs(seed_means[0.60][i])+abs(seed_means[0.80][i]))/2 for i in range(n)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    ax = axes[0]
    ax.fill_between(PX, [m-s for m,s in zip(se_means,se_sems)],
                    [m+s for m,s in zip(se_means,se_sems)], color='#E6550D', alpha=0.15)
    ax.errorbar(PX, se_means, yerr=se_sems, fmt='o-', color='#E6550D', lw=1.5, ms=5, capsize=3,
                label=f'TCM (n={n})')
    if rnn_se_means is not None:
        ax.errorbar(PX, rnn_se_means, yerr=rnn_se_sems, fmt='o-', color='gray',
                    lw=1.5, ms=5, capsize=3, label='RNN')
    ax.fill_between(PX, [m-s for m,s in zip(hs,hs_s)],
                    [m+s for m,s in zip(hs,hs_s)], color='black', alpha=0.10)
    ax.errorbar(PX, hs, yerr=hs_s, fmt='s-', color='black', ms=5, lw=1.5, capsize=3,
                label='Human')
    ax.set_xlabel('True Position (%)'); ax.set_ylabel('Signed Error (%)')
    ax.legend(fontsize=8); ax.set_xticks(PX)

    ax = axes[1]
    ax.bar([0,1], [np.mean(seed_early), np.mean(seed_late)],
           color=['#D73027','#4575B4'], edgecolor='white', lw=1, width=0.5)
    ax.errorbar([0,1], [np.mean(seed_early), np.mean(seed_late)],
                yerr=[np.std(seed_early,ddof=1)/np.sqrt(n),
                      np.std(seed_late,ddof=1)/np.sqrt(n)],
                fmt='none', color='black', capsize=5, lw=1)
    ax.set_xticks([0,1]); ax.set_xticklabels(['20%+40%','60%+80%'])

    ax = axes[2]
    x_all = np.concatenate([all_errors[pf] for pf in POSITIONS])
    xs = np.linspace(x_all.min()-5, x_all.max()+5, 200)
    for i, pf in enumerate(POSITIONS):
        kde = gaussian_kde(np.array(all_errors[pf]))
        ax.fill_between(xs, kde(xs), alpha=0.3, color=rdylbu_4[i])
        ax.plot(xs, kde(xs), color=rdylbu_4[i], lw=1.5, label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('Density')

    ax = axes[3]
    for i, pf in enumerate(POSITIONS):
        data = np.sort(all_errors[pf])
        ax.plot(data, np.arange(1,len(data)+1)/len(data),
                color=rdylbu_4[i], lw=1.5, label=f'{int(pf*100)}%')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Signed Error (%)'); ax.set_ylabel('CDF')
    ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'tcm_rho{rho:.2f}_panels_signed.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  tcm_rho{rho:.2f}_panels_signed.png  (asym={asym:+.1f}, n={n})")

# ── Supplementary: μ across rho ───────────────────────────
print("\nGenerating supp figures...")

model_mu_data = {}
model_se_data = {}
for rho in RHYLOS:
    mm = load_model_mu(LOGS, rho)
    if mm[0.20]:
        model_mu_data[rho] = {
            'means': [np.mean(mm[pf]) for pf in POSITIONS],
            'sems': [np.std(mm[pf],ddof=1)/np.sqrt(max(len(mm[pf]),1))
                     for pf in POSITIONS],
            'n': len(mm[0.20]),
        }
    sm, _ = load_seed_means(LOGS, rho)
    if sm[0.20]:
        model_se_data[rho] = {
            'means': [np.mean(sm[pf]) for pf in POSITIONS],
            'sems': [np.std(sm[pf],ddof=1)/np.sqrt(len(sm[pf]))
                     for pf in POSITIONS] if sm[0.20] else [0]*4,
            'n': len(sm[0.20]),
        }


def make_supp(title, ylabel, human_means, human_sems, model_data, outname,
               rnn_m=None, rnn_s=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(PX, [m-s for m,s in zip(human_means,human_sems)],
                    [m+s for m,s in zip(human_means,human_sems)],
                    color='black', alpha=0.10)
    ax.errorbar(PX, human_means, yerr=human_sems, fmt='s-',
                color='black', ms=8, lw=2, capsize=4, label='Human')
    for rho in RHYLOS:
        if rho not in model_data: continue
        md = model_data[rho]
        c = viridis_rho[rho]
        ax.fill_between(PX, [m-s for m,s in zip(md['means'],md['sems'])],
                        [m+s for m,s in zip(md['means'],md['sems'])],
                        color=c, alpha=0.12)
        ax.errorbar(PX, md['means'], yerr=md['sems'], fmt='o-',
                    color=c, ms=6, lw=1.5, capsize=3,
                    label=f'TCM ρ={rho:.2f} (n={md["n"]})')
    if rnn_m is not None:
        ax.errorbar(PX, rnn_m, yerr=rnn_s if rnn_s else None,
                    fmt='D--', color='gray', ms=6, lw=1.5, capsize=3,
                    label='RNN')
    ax.axhline(0, color='black', lw=0.6, ls=':', alpha=0.4)
    ax.set_xlabel('Temporal Location', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(PX); ax.set_xticklabels(['20%','40%','60%','80%'])
    ax.set_xlim(12, 88)
    ax.legend(fontsize=9, frameon=True, loc='lower left', ncol=2)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25, linestyle='--')
    fig.tight_layout()
    png = os.path.join(OUT, outname)
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(png.replace('.png','.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  {outname}")

make_supp('A: MemToolbox-estimated μ by temporal location', 'μ',
          hm, hm_s, model_mu_data, 'supp_mu_across_rho.png',
          rnn_m=rnn_mu_means, rnn_s=rnn_mu_sems)
make_supp('B: Signed error by temporal location', 'Signed Error (%)',
          hs, hs_s, model_se_data, 'supp_signed_error_across_rho.png',
          rnn_m=rnn_se_means, rnn_s=rnn_se_sems)

print(f"\nDone. All figures in {OUT}/")
