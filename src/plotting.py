"""
Plotting Utilities
===================
Generate figures from saved evaluation data.
No model training required — reads from .npz files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

from .config import TEST_POSITIONS, HUMAN_ERRORS
from .analysis import compute_summary_stats, compute_human_loss


# Color schemes
RHO_COLORS = ['#e41a1c', '#ff7f00', '#377eb8', '#4daf4a', '#984ea3', '#888888']
POS_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
POS_LABELS = ['20%', '40%', '60%', '80%']


def plot_error_curve(errors_dict, rho, sigma_m, metadata=None,
                     human_compare=True, save_path=None):
    """Error-vs-position curve with slope annotation."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    x_pos = [20, 40, 60, 80]
    means, sems, asymmetry = compute_summary_stats(errors_dict)

    # Fit slope
    x_vals = np.array(TEST_POSITIONS) * 100
    slope, intercept, r_val, _, _ = stats.linregress(x_vals, means)

    # Plot model
    ax.errorbar(x_pos, means, yerr=sems, fmt='o-', color='#e41a1c',
                linewidth=2.5, markersize=9, capsize=5, label='Model', zorder=3)
    x_fit = np.linspace(10, 90, 100)
    ax.plot(x_fit, slope * x_fit + intercept, '--', color='#e41a1c',
            linewidth=1, alpha=0.5)

    # Plot human comparison
    if human_compare:
        ax.plot(x_pos, HUMAN_ERRORS, 'o-', color='black',
                linewidth=2, markersize=9, label='Human (Hu et al.)', zorder=2)

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('True Position (%)', fontsize=11)
    ax.set_ylabel('Signed Error (%)', fontsize=11)
    ax.set_title(
        f'TCM Model: ρ={rho:.2f}, σ_m={sigma_m:.2f}\n'
        f'slope β={slope:.3f}, R²={r_val**2:.3f}, '
        f'asymmetry={asymmetry:+.1f}%',
        fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(10, 90)

    plt.tight_layout()
    if save_path:
        ensure_dir_save(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_error_distributions(errors_dict, rho, sigma_m, save_path=None):
    """Violin plot of error distributions at each position."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    for j, pf in enumerate(TEST_POSITIONS):
        vp = ax.violinplot(errors_dict[pf], positions=[j], vert=True,
                           showmeans=True, showmedians=True, widths=0.7)
        for body in vp['bodies']:
            body.set_facecolor(POS_COLORS[j])
            body.set_alpha(0.5)
        for part in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if part in vp:
                vp[part].set_color(POS_COLORS[j])

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(POS_LABELS)
    ax.set_ylabel('Signed Error (%)', fontsize=11)
    ax.set_title(f'Error Distributions  (ρ={rho:.2f}, σ_m={sigma_m:.2f})',
                 fontsize=10)
    plt.tight_layout()

    if save_path:
        ensure_dir_save(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_attention_weights(attn_data, rho, sigma_m, seq_test=100, save_path=None):
    """Attention weight profiles showing forward skew mechanism."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    x = np.arange(seq_test)

    for j, pf in enumerate(TEST_POSITIONS):
        if pf not in attn_data:
            continue
        w = attn_data[pf]['weights']
        ti = attn_data[pf]['true_idx']
        pred = attn_data[pf]['pred']
        ax.plot(x, w, color=POS_COLORS[j], linewidth=1.3, alpha=0.75,
                label=f'{POS_LABELS[j]} (pred={pred:.0f}%)')
        ax.axvline(ti, color=POS_COLORS[j], linestyle='--',
                   linewidth=0.6, alpha=0.4)

    ax.set_xlabel('Position index', fontsize=11)
    ax.set_ylabel('Attention weight', fontsize=11)
    ax.set_title(f'Attention Weights — Forward Skew Mechanism  (ρ={rho:.2f})',
                 fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()

    if save_path:
        ensure_dir_save(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_sweep_summary(all_run_data, save_path=None):
    """
    Plot sweep summary: asymmetry vs rho, error curves for all models,
    and loss landscape.

    all_run_data: list of dicts with keys:
        rho, sigma_m, errors_dict, loss (sum|human - model|)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    x_pos = [20, 40, 60, 80]

    # Collect unique rhos/sigmas
    all_rho = sorted(set(d['rho'] for d in all_run_data))
    all_sigma = sorted(set(d['sigma_m'] for d in all_run_data))

    # --- A: Loss heatmap ---
    ax = axes[0, 0]
    loss_grid = np.full((len(all_rho), len(all_sigma)), np.nan)
    for d in all_run_data:
        i = all_rho.index(d['rho'])
        j = all_sigma.index(d['sigma_m'])
        loss_grid[i, j] = d.get('loss', 0)

    im = ax.imshow(loss_grid, aspect='auto', cmap='Reds_r', origin='lower',
                   extent=[all_sigma[0] - 0.025, all_sigma[-1] + 0.025,
                           all_rho[0] - 0.025, all_rho[-1] + 0.025])
    ax.set_xticks(all_sigma)
    ax.set_yticks(all_rho)
    ax.set_xlabel('σ_measurement', fontsize=11)
    ax.set_ylabel('ρ', fontsize=11)
    ax.set_title('A  Loss = Σ|Human − Model|', fontsize=11,
                 fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Best
    if all_run_data:
        best = min(all_run_data, key=lambda d: d.get('loss', 1e6))
        ax.scatter(best['sigma_m'], best['rho'], marker='*', s=300,
                   color='#2ca02c', edgecolors='black', linewidth=1, zorder=5)

    # --- B: All error curves vs human ---
    ax = axes[0, 1]
    ax.plot(x_pos, HUMAN_ERRORS, 'o-', color='black', linewidth=2.5,
            markersize=10, label='Human', zorder=5)
    for d in all_run_data:
        means = [np.mean(d['errors_dict'][p]) for p in TEST_POSITIONS]
        ax.plot(x_pos, means, 's--', linewidth=1, markersize=4,
                alpha=0.5, label=f"ρ={d['rho']:.2f},σ={d['sigma_m']:.2f}")
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.4)
    ax.set_xlabel('True Position (%)')
    ax.set_ylabel('Signed Error (%)')
    ax.set_title('B  All Model Curves vs. Human', fontsize=11,
                 fontweight='bold', loc='left')
    if len(all_run_data) <= 12:
        ax.legend(fontsize=7)

    # --- C: Asymmetry vs rho ---
    ax = axes[1, 0]
    for j, sm in enumerate(all_sigma):
        rhos_here = []; asms_here = []
        for d in all_run_data:
            if d['sigma_m'] == sm:
                rhos_here.append(d['rho'])
                means = [np.mean(d['errors_dict'][p]) for p in TEST_POSITIONS]
                asm = (abs(means[0])+abs(means[1])-abs(means[2])-abs(means[3]))/2
                asms_here.append(asm)
        if rhos_here:
            order = np.argsort(rhos_here)
            ax.plot(np.array(rhos_here)[order], np.array(asms_here)[order],
                    'o-', linewidth=1.5, markersize=6,
                    label=f'σ_m={sm:.2f}')
    human_early = (abs(HUMAN_ERRORS[0])+abs(HUMAN_ERRORS[1]))/2
    human_late = (abs(HUMAN_ERRORS[2])+abs(HUMAN_ERRORS[3]))/2
    ax.axhline(human_early-human_late, color='black', linestyle=':',
               linewidth=1.5, label=f'Human: {human_early-human_late:.1f}')
    ax.set_xlabel('ρ', fontsize=11)
    ax.set_ylabel('Asymmetry (%)', fontsize=11)
    ax.set_title('C  Asymmetry vs. ρ by σ_m', fontsize=11,
                 fontweight='bold', loc='left')
    ax.legend(fontsize=8)

    # --- D: Slope vs rho ---
    ax = axes[1, 1]
    for j, sm in enumerate(all_sigma):
        rhos_here = []; slopes_here = []
        for d in all_run_data:
            if d['sigma_m'] == sm:
                rhos_here.append(d['rho'])
                means = [np.mean(d['errors_dict'][p]) for p in TEST_POSITIONS]
                xv = np.array(TEST_POSITIONS) * 100
                sl, _, _, _, _ = stats.linregress(xv, means)
                slopes_here.append(sl)
        if rhos_here:
            order = np.argsort(rhos_here)
            ax.plot(np.array(rhos_here)[order], np.array(slopes_here)[order],
                    'o-', linewidth=1.5, markersize=6,
                    label=f'σ_m={sm:.2f}')
    ax.axhline(-0.117, color='black', linestyle=':', linewidth=1.5,
               label='Human β=−0.117')
    ax.set_xlabel('ρ', fontsize=11)
    ax.set_ylabel('Slope β', fontsize=11)
    ax.set_title('D  Error-Position Slope vs. ρ by σ_m', fontsize=11,
                 fontweight='bold', loc='left')
    ax.legend(fontsize=8)

    plt.suptitle('TCM Model — Parameter Sweep Summary', fontsize=14,
                 fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        ensure_dir_save(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def ensure_dir_save(file_path):
    """Create parent directory for a file path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
