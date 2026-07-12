#!/usr/bin/env python3
"""Plot the long-sequence (150--180) TCM results: bias curves vs human.

Run ``python train_long_seq.py`` first.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import HUMAN_ERRORS, TEST_POSITIONS
from src.utils import discover_runs, load_results, parse_run_path

# ── Constants ─────────────────────────────────────────────────────────
RHO = 0.95
SIGMA = 0.05
D_CONTEXT = 64
SEEDS = [42, 43, 44, 45, 46]
LOGS_ROOT = os.path.join(os.path.dirname(__file__), "logs_longseq")
FIGURE_PATH = os.path.join(
    os.path.dirname(__file__), "figures", "tcm_long_seq.png")

# ── Style ─────────────────────────────────────────────────────────────
POS_COLORS = ["#1F78B4", "#33A02C", "#E31A1C", "#6A3D9A"]
HUMAN_BLACK = "#222222"
MODEL_RED = "#E31A1C"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.fontsize": 8,
    "legend.frameon": False,
})


# ── Helpers ───────────────────────────────────────────────────────────

def find_longseq_runs():
    """Locate logs_longseq/ runs matching the expected (rho, sigma, seeds)."""
    runs = {}
    for run_path in discover_runs(LOGS_ROOT):
        info = parse_run_path(run_path)
        if (abs(info["rho"] - RHO) < 0.005
                and abs(info["sigma_m"] - SIGMA) < 0.005
                and info["d"] == D_CONTEXT
                and info["seed"] in SEEDS):
            runs[info["seed"]] = run_path
    missing = [s for s in SEEDS if s not in runs]
    if missing:
        raise RuntimeError(
            f"Missing long-seq logs for seeds {missing}. "
            "Run `python train_long_seq.py` first.")
    return [runs[s] for s in SEEDS]


def load_behavior(run_paths):
    """Seed-level means + pooled trial distributions."""
    seed_means = []
    pooled = {pf: [] for pf in TEST_POSITIONS}
    asymmetries = []
    for rp in run_paths:
        errors, _, _ = load_results(rp)
        means = np.array([np.mean(errors[pf]) for pf in TEST_POSITIONS])
        seed_means.append(means)
        asymmetries.append(
            (abs(means[0]) + abs(means[1])
             - abs(means[2]) - abs(means[3])) / 2)
        for pf in TEST_POSITIONS:
            pooled[pf].extend(errors[pf].tolist())
    return (
        np.stack(seed_means),
        {pf: np.asarray(v) for pf, v in pooled.items()},
        np.asarray(asymmetries),
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    run_paths = find_longseq_runs()
    seed_means, pooled_errors, asymmetries = load_behavior(run_paths)

    mean_errors = seed_means.mean(axis=0)
    sd_errors = seed_means.std(axis=0)
    human_asymmetry = (
        abs(HUMAN_ERRORS[0]) + abs(HUMAN_ERRORS[1])
        - abs(HUMAN_ERRORS[2]) - abs(HUMAN_ERRORS[3])) / 2

    # Build figure
    fig = plt.figure(figsize=(15, 5.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.34)

    test_percent = np.asarray(TEST_POSITIONS) * 100

    # ── A: Error curve with seed band ─────────────────────────────
    ax = fig.add_subplot(grid[0])
    ax.fill_between(
        test_percent, mean_errors - sd_errors, mean_errors + sd_errors,
        color=MODEL_RED, alpha=0.2, linewidth=0)
    ax.plot(test_percent, mean_errors, "o-", color=MODEL_RED, linewidth=2.5,
            markersize=9, markeredgecolor="white", markeredgewidth=0.5,
            label="TCM (mean ± 1SD, 5 seeds)")

    # Individual seed traces (faint)
    for i in range(len(SEEDS)):
        ax.plot(test_percent, seed_means[i], ".--", color=MODEL_RED,
                linewidth=0.7, markersize=4, alpha=0.35)

    ax.plot(test_percent, HUMAN_ERRORS, "D-", color=HUMAN_BLACK,
            linewidth=2.5, markersize=9, label="Human (Hu et al.)", zorder=5)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set(xlabel="True position (%)", ylabel="Signed error (%)",
           xlim=(10, 90))
    ax.set_title(
        "A  Error Pattern — Long sequences (150–180)",
        fontweight="bold", loc="left")
    ax.legend(fontsize=8)

    # ── B: Pooled error distributions ────────────────────────────
    ax = fig.add_subplot(grid[1])
    x_range = np.linspace(-50, 50, 500)
    bins = np.linspace(-50, 50, 81)
    for pi, pf in enumerate(TEST_POSITIONS):
        values = pooled_errors[pf]
        ax.hist(values, bins=bins, density=True, alpha=0.16,
                color=POS_COLORS[pi], edgecolor=POS_COLORS[pi],
                linewidth=0.3)
        kde = gaussian_kde(values, bw_method="scott")
        ax.plot(x_range, kde(x_range), color=POS_COLORS[pi],
                linewidth=2.1,
                label=f"{int(pf*100)}% (mean={np.mean(values):+.1f}%)")
        ax.axvline(np.mean(values), color=POS_COLORS[pi],
                   linestyle="--", linewidth=1, alpha=0.45)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set(xlabel="Signed error (%)", ylabel="Probability density",
           xlim=(-45, 45))
    ax.set_title(
        "B  Error Distributions (5 seeds pooled)",
        fontweight="bold", loc="left")
    ax.legend(fontsize=8)

    # ── C: Summary table ─────────────────────────────────────────
    ax = fig.add_subplot(grid[2])
    ax.axis("off")

    summary_lines = [
        "TCM — Long Training Sequences (150–180)",
        "═" * 42,
        f"  ρ = {RHO:.2f}     σₘ = {SIGMA:.2f}     LR = 5e-2     epochs = 4000",
        "",
        "  Seed          Errors [20%, 40%, 60%, 80%]          Asymmetry",
        "  ────   ───────────────────────────────────────   ──────────",
    ]
    for i, s in enumerate(SEEDS):
        errs_str = "  ".join(f"{v:+6.2f}" for v in seed_means[i])
        summary_lines.append(
            f"  {s:3d}     {errs_str}              {asymmetries[i]:+6.2f}")
    summary_lines.extend([
        "",
        f"  Mean     {mean_errors[0]:+6.2f}    {mean_errors[1]:+6.2f}"
        f"    {mean_errors[2]:+6.2f}    {mean_errors[3]:+6.2f}"
        f"              {asymmetries.mean():+6.2f}",
        f"  ±1SD     {sd_errors[0]:6.2f}    {sd_errors[1]:6.2f}"
        f"    {sd_errors[2]:6.2f}    {sd_errors[3]:6.2f}"
        f"              {asymmetries.std():6.2f}",
        "",
        f"  Human    {HUMAN_ERRORS[0]:+6.2f}    {HUMAN_ERRORS[1]:+6.2f}"
        f"    {HUMAN_ERRORS[2]:+6.2f}    {HUMAN_ERRORS[3]:+6.2f}"
        f"              {human_asymmetry:+6.2f}",
        "",
        f"  Human loss (Σ|model − human|) = {np.sum(abs(mean_errors - np.array(HUMAN_ERRORS))):.2f}",
    ])

    text = "\n".join(summary_lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=8, fontfamily="monospace",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", linewidth=0.5))
    ax.set_title("C  Summary", fontweight="bold", loc="left",
                 fontsize=12)

    fig.suptitle(
        "TCM — Long Training Sequences (150–180)  |  "
        f"asymmetry = {asymmetries.mean():+.1f} ± "
        f"{asymmetries.std():.1f}  (human = {human_asymmetry:+.1f})",
        fontsize=13, fontweight="bold", y=0.995)

    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
