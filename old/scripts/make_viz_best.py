#!/usr/bin/env python3
"""Final behavioral figure for the best five-seed TCM ensemble.

Panels A-B summarize behavior. Panel F decomposes the trained model's signed
position estimate into forward and backward distance-weighted attention.

Run ``python train_best_models.py`` once before running this script. The
training command saves the checkpoints that the original summary logs lacked.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import D_ITEM, HUMAN_ERRORS, SEQ_TEST, TEST_POSITIONS
from src.model import AttentionDecoder, TCMEncoder
from src.utils import discover_runs, load_results, parse_run_path


RHO = 0.95
SIGMA = 0.05
D_CONTEXT = 64
SEEDS = [42, 43, 44, 45, 46]
LOGS_ROOT = os.path.join(os.path.dirname(__file__), "logs_best")
RNN_LOGS_ROOT = os.path.join(os.path.dirname(__file__), "logs_rnn")
FIGURE_PATH = os.path.join(
    os.path.dirname(__file__), "figures", "tcm_viz_best_model.png")

POS_COLORS_LIGHT = ["#A6CEE3", "#B2DF8A", "#FB9A99", "#CAB2D6"]
POS_COLORS = ["#1F78B4", "#33A02C", "#E31A1C", "#6A3D9A"]
HUMAN_BLACK = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.fontsize": 7.5,
    "legend.frameon": False,
})


def best_run_paths():
    """Return the five expected run paths in seed order."""
    runs = {}
    for run_path in discover_runs(LOGS_ROOT):
        info = parse_run_path(run_path)
        if (abs(info["rho"] - RHO) < 0.005
                and abs(info["sigma_m"] - SIGMA) < 0.005
                and info["d"] == D_CONTEXT
                and info["seed"] in SEEDS):
            runs[info["seed"]] = run_path
    missing = [seed for seed in SEEDS if seed not in runs]
    if missing:
        raise RuntimeError(f"Missing best-model logs for seeds: {missing}")
    return [runs[seed] for seed in SEEDS]


def load_behavior(run_paths):
    """Load seed-level error means and pooled trial-level distributions."""
    seed_means = []
    pooled = {pf: [] for pf in TEST_POSITIONS}
    asymmetries = []
    for run_path in run_paths:
        errors, _, _ = load_results(run_path)
        means = np.array([np.mean(errors[pf]) for pf in TEST_POSITIONS])
        seed_means.append(means)
        asymmetries.append(
            (abs(means[0]) + abs(means[1])
             - abs(means[2]) - abs(means[3])) / 2)
        for pf in TEST_POSITIONS:
            pooled[pf].extend(errors[pf].tolist())
    return (
        np.stack(seed_means),
        {pf: np.asarray(values) for pf, values in pooled.items()},
        np.asarray(asymmetries),
    )


def load_rnn_seed_means():
    """Load the five-seed simple-RNN control used in panel A."""
    by_seed = {}
    for run_path in discover_runs(RNN_LOGS_ROOT):
        info = parse_run_path(run_path)
        if info["seed"] in SEEDS:
            errors, _, _ = load_results(run_path)
            by_seed[info["seed"]] = np.array([
                np.mean(errors[pf]) for pf in TEST_POSITIONS])
    missing = [seed for seed in SEEDS if seed not in by_seed]
    if missing:
        raise RuntimeError(
            f"Missing simple-RNN results for seeds {missing}. "
            "Run `python train_rnn_models.py` first.")
    return np.stack([by_seed[seed] for seed in SEEDS])


def load_checkpoint(run_path):
    checkpoint_path = os.path.join(run_path, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f"Missing {checkpoint_path}. Run `python train_best_models.py` first.")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    encoder = TCMEncoder(
        checkpoint["rho"], checkpoint["d_item"], checkpoint["d_context"],
        use_proj=True)
    decoder = AttentionDecoder(
        checkpoint["d_context"], use_position_template=True)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def diagnose_model(encoder, decoder, n_sequences, batch_size, eval_seed):
    """Average readout contributions over a fixed bank of random sequences."""
    length = SEQ_TEST
    generator = torch.Generator(device="cpu")
    generator.manual_seed(eval_seed)

    forward_sum = torch.zeros(length)
    backward_sum = torch.zeros(length)
    net_error_sum = torch.zeros(length)
    seen = 0

    template = torch.linspace(0, 1, length)
    query_target = torch.arange(length).float() / length
    displacement = template.unsqueeze(0) - template.unsqueeze(1)
    forward_distance = displacement.clamp_min(0)
    backward_distance = (-displacement).clamp_min(0)
    template_offset = template - query_target

    with torch.no_grad():
        while seen < n_sequences:
            current_batch = min(batch_size, n_sequences - seen)
            items = torch.randn(
                current_batch, length, D_ITEM, generator=generator)
            _, contexts = encoder(items)

            noise = SIGMA * torch.randn(
                contexts.shape, generator=generator,
                dtype=contexts.dtype, device=contexts.device)
            noisy_queries = F.normalize(contexts + noise, dim=-1)
            keys = decoder.key(contexts)
            queries = decoder.query(noisy_queries)
            logits = torch.bmm(queries, keys.transpose(1, 2)) / decoder.temperature
            weights = F.softmax(logits, dim=-1)

            forward = (weights * forward_distance).sum(dim=-1)
            backward = (weights * backward_distance).sum(dim=-1)
            net_error = forward - backward + template_offset.unsqueeze(0)

            forward_sum += forward.sum(dim=0)
            backward_sum += backward.sum(dim=0)
            net_error_sum += net_error.sum(dim=0)
            seen += current_batch

    return {
        "forward": (100 * forward_sum / seen).numpy(),
        "backward": (100 * backward_sum / seen).numpy(),
        "net_error": (100 * net_error_sum / seen).numpy(),
        "temperature": float(decoder.temperature.item()),
    }


def collect_diagnostics(run_paths, n_sequences, batch_size):
    diagnostics = []
    for seed, run_path in zip(SEEDS, run_paths):
        print(f"Diagnosing trained model seed={seed}...", flush=True)
        encoder, decoder = load_checkpoint(run_path)
        # Resetting the generator for each seed gives every model the same
        # sequences and retrieval-noise draws, making seed contrasts paired.
        diagnostics.append(diagnose_model(
            encoder, decoder, n_sequences, batch_size,
            eval_seed=20260623))
    return diagnostics


def stack(diagnostics, key):
    return np.stack([diagnostic[key] for diagnostic in diagnostics])


def add_seed_band(ax, x, values, color, label, linestyle="-"):
    mean = values.mean(axis=0)
    sd = values.std(axis=0)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.16,
                    linewidth=0)
    ax.plot(x, mean, linestyle, color=color, linewidth=2, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-sequences", type=int, default=200,
                        help="sequences per trained seed for mechanism panels")
    parser.add_argument("--diagnostic-batch", type=int, default=50)
    args = parser.parse_args()

    run_paths = best_run_paths()
    seed_means, pooled_errors, asymmetries = load_behavior(run_paths)
    rnn_seed_means = load_rnn_seed_means()
    diagnostics = collect_diagnostics(
        run_paths, args.diagnostic_sequences, args.diagnostic_batch)

    mean_errors = seed_means.mean(axis=0)
    sd_errors = seed_means.std(axis=0)
    rnn_mean_errors = rnn_seed_means.mean(axis=0)
    rnn_sd_errors = rnn_seed_means.std(axis=0)
    human_asymmetry = (
        abs(HUMAN_ERRORS[0]) + abs(HUMAN_ERRORS[1])
        - abs(HUMAN_ERRORS[2]) - abs(HUMAN_ERRORS[3])) / 2

    fig = plt.figure(figsize=(19, 5.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.34)
    position_percent = np.arange(SEQ_TEST) / SEQ_TEST * 100

    # A: behavioral result.
    ax = fig.add_subplot(grid[0])
    test_percent = np.asarray(TEST_POSITIONS) * 100
    ax.fill_between(
        test_percent, mean_errors - sd_errors, mean_errors + sd_errors,
        color="#E31A1C", alpha=0.2, linewidth=0)
    ax.plot(test_percent, mean_errors, "o-", color="#E31A1C", linewidth=2.5,
            markersize=9, markeredgecolor="white", markeredgewidth=0.5,
            label="TCM (mean ± SD)")
    ax.fill_between(
        test_percent, rnn_mean_errors - rnn_sd_errors,
        rnn_mean_errors + rnn_sd_errors,
        color="#1F78B4", alpha=0.16, linewidth=0)
    ax.plot(test_percent, rnn_mean_errors, "s--", color="#1F78B4",
            linewidth=2.1, markersize=7, markeredgecolor="white",
            markeredgewidth=0.5, label="Simple RNN (mean ± SD)")
    ax.plot(test_percent, HUMAN_ERRORS, "D-", color=HUMAN_BLACK,
            linewidth=2.5, markersize=9, label="Human (Hu et al.)", zorder=5)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set(xlabel="True position (%)", ylabel="Signed error (%)", xlim=(10, 90))
    ax.set_title("A  Error Pattern (5 trained seeds)", fontweight="bold", loc="left")
    ax.legend(fontsize=8)

    # B: pooled behavioral distributions (descriptive; uncertainty is in A).
    ax = fig.add_subplot(grid[1])
    x_range = np.linspace(-50, 50, 500)
    bins = np.linspace(-50, 50, 81)
    for panel_idx, pf in enumerate(TEST_POSITIONS):
        values = pooled_errors[pf]
        ax.hist(values, bins=bins, density=True, alpha=0.16,
                color=POS_COLORS[panel_idx], edgecolor=POS_COLORS[panel_idx],
                linewidth=0.3)
        kde = gaussian_kde(values, bw_method="scott")
        ax.plot(x_range, kde(x_range), color=POS_COLORS[panel_idx],
                linewidth=2.1,
                label=f"{int(pf * 100)}% (mean={np.mean(values):+.1f}%)")
        ax.axvline(np.mean(values), color=POS_COLORS[panel_idx],
                   linestyle="--", linewidth=1, alpha=0.45)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set(xlabel="Signed error (%)", ylabel="Probability density", xlim=(-45, 45))
    ax.set_title("B  Error Distributions (5 seeds pooled)", fontweight="bold", loc="left")
    ax.legend(fontsize=8)

    # F: exact readout decomposition. Forward and backward contributions are
    # relative to the template entry at i; net also includes the tiny i/(L-1)
    # versus i/L template offset used by the original training target.
    ax = fig.add_subplot(grid[2])
    forward = stack(diagnostics, "forward")
    backward = stack(diagnostics, "backward")
    net_error = stack(diagnostics, "net_error")
    add_seed_band(ax, position_percent, forward, "#E6550D", "forward contribution")
    add_seed_band(ax, position_percent, -backward, "#3182BD", "backward contribution")
    add_seed_band(ax, position_percent, net_error, "#222222", "net signed error")
    ax.axhline(0, color="gray", linewidth=0.8)
    for pf in TEST_POSITIONS:
        ax.axvline(100 * pf, color="gray", linestyle=":", linewidth=0.6,
                   alpha=0.35)
    ax.set(xlabel="Query position (%)", ylabel="Contribution to estimate (%)",
           xlim=(0, 99))
    ax.set_title("C  Forward/Backward Readout Contributions", fontweight="bold", loc="left")
    ax.legend(fontsize=7.5)

    mean_temperature = np.mean([d["temperature"] for d in diagnostics])
    fig.suptitle(
        "TCM Best Model — trained P/K/Q diagnostics  |  "
        f"ρ={RHO}, σ={SIGMA}, LR=5e-2, 4K epochs, 5 seeds  |  "
        f"asymmetry={asymmetries.mean():+.1f} (human {human_asymmetry:+.1f}), "
        f"T={mean_temperature:.3f}",
        fontsize=13, fontweight="bold", y=0.995)

    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
