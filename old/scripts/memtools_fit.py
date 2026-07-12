#!/usr/bin/env python3
"""
Fit model trial-level outputs through a MemToolbox-equivalent mixture model.

MemToolbox Orientation(WithBias(StandardMixtureModel)) fits:
    p(response) = (1 - g) * VM(response | true + mu, kappa) + g * U(0, 2pi)

Compares model-derived mu against human mu from Hu et al.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import i0 as bessel_i0
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import TCMEncoder, AttentionDecoder
from src.utils import apply_noise, load_results, param_dir
from src.config import (
    D_ITEM, D_CONTEXT, SEQ_TEST, TEST_POSITIONS, HUMAN_ERRORS,
)

# ── Best model configuration ──────────────────────────────────────────
RHO = 0.95
SIGMA_M = 0.05
SEEDS = [42, 43, 44, 45, 46]
N_TRIALS_PER_POS = 2000
BATCH = 200
CHECKPOINT_BASE = os.path.join(os.path.dirname(__file__), "logs_best")
RESULTS_BASE = os.path.join(os.path.dirname(__file__), "logs_best")

# ── Style matching make_viz_best.py ────────────────────────────────────
HUMAN_BLACK = "#222222"
MODEL_RED = "#E31A1C"
MODEL_RED_LIGHT = "#FCBBA1"
MEMTOOLS_BLUE = "#1F78B4"
MEMTOOLS_BLUE_LIGHT = "#A6CEE3"
RAW_PURPLE = "#6A3D9A"
RAW_PURPLE_LIGHT = "#CAB2D6"
KAPPA_GREEN = "#33A02C"
KAPPA_GREEN_LIGHT = "#B2DF8A"
GUESS_ORANGE = "#FF7F00"
GUESS_ORANGE_LIGHT = "#FDBF6F"

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
    "legend.fontsize": 8,
    "legend.frameon": False,
})


# ── Load checkpoint ───────────────────────────────────────────────────
def load_model_from_checkpoint(seed):
    run_dir = param_dir(RHO, SIGMA_M, D_CONTEXT, seed, base=CHECKPOINT_BASE)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    encoder = TCMEncoder(state["rho"], state["d_item"],
                         state["d_context"], use_proj=True)
    decoder = AttentionDecoder(state["d_context"], use_position_template=True)
    encoder.load_state_dict(state["encoder_state_dict"])
    decoder.load_state_dict(state["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder


# ── Sample trial-level outputs ────────────────────────────────────────
def sample_trials(encoder, decoder, n_trials=N_TRIALS_PER_POS, batch=BATCH):
    L = SEQ_TEST
    outputs = {pf: [] for pf in TEST_POSITIONS}
    with torch.no_grad():
        for pf in TEST_POSITIONS:
            ti = int(pf * L)
            for _ in range(n_trials // batch):
                items = torch.randn(batch, L, D_ITEM)
                f, c = encoder(items)
                c_q = apply_noise(c[:, ti, :], SIGMA_M)
                pred = decoder(c_q, c).cpu().numpy()
                outputs[pf].extend(pred.tolist())
    return {pf: np.array(v) for pf, v in outputs.items()}


# ── MemToolbox-equivalent von Mises mixture fit ───────────────────────
def _wrap_to_circle(x, target, period=1.0):
    return target + ((x - target + period/2) % period) - period/2


def _vm_logpdf(x, mu, kappa):
    return kappa * np.cos(x - mu) - np.log(2 * np.pi * bessel_i0(kappa))


def _nll(params, x_rad, true_rad):
    mu_bias = params[0]
    kappa = np.exp(params[1])
    g = 1.0 / (1.0 + np.exp(-params[2]))
    mu_biased = true_rad + mu_bias
    ll_vm = _vm_logpdf(x_rad, mu_biased, kappa)
    ll_u = -np.log(2 * np.pi)
    log_p = np.logaddexp(np.log(1 - g) + ll_vm, np.log(g) + ll_u)
    return -np.sum(log_p)


def _fit_single_position(x_raw, true_val, n_restarts=10):
    x_rad = x_raw * 2 * np.pi
    true_rad = true_val * 2 * np.pi
    x_rad_wrapped = _wrap_to_circle(x_rad, true_rad, period=2 * np.pi)

    best_nll = np.inf
    best_params = None
    for _ in range(n_restarts):
        mu_init = np.random.uniform(-0.5, 0.5)
        kappa_init = np.exp(np.random.uniform(1.0, 4.0))
        g_init = np.random.beta(0.5, 5.0)
        logit_g_init = np.log(g_init / (1 - g_init))
        log_kappa_init = np.log(kappa_init)
        init = np.array([mu_init, log_kappa_init, logit_g_init])
        res = minimize(
            _nll, init, args=(x_rad_wrapped, true_rad),
            method="L-BFGS-B",
            bounds=[
                (-np.pi, np.pi),
                (np.log(0.1), np.log(500)),
                (-10, 5),
            ],
        )
        if res.fun < best_nll:
            best_nll = res.fun
            best_params = res.x

    mu_rad, log_kappa, logit_g = best_params
    return {
        "mu_pct": (mu_rad / (2 * np.pi)) * 100,
        "kappa": np.exp(log_kappa),
        "g": 1.0 / (1.0 + np.exp(-logit_g)),
        "nll": best_nll,
    }


def fit_memtools_model(all_outputs):
    per_seed = []
    all_mu, all_kappa, all_g = [], [], []
    for seed_idx, outputs in enumerate(all_outputs):
        seed_fits = {}
        mu_row, kap_row, g_row = [], [], []
        for pf in TEST_POSITIONS:
            fit = _fit_single_position(outputs[pf], pf, n_restarts=10)
            seed_fits[pf] = fit
            mu_row.append(fit["mu_pct"])
            kap_row.append(fit["kappa"])
            g_row.append(fit["g"])
            print(f"  seed={SEEDS[seed_idx]} pos={pf:.0%}: "
                  f"mu={fit['mu_pct']:+.3f}%  kappa={fit['kappa']:.1f}  "
                  f"g={fit['g']:.4f}  NLL={fit['nll']:.1f}")
        per_seed.append(seed_fits)
        all_mu.append(mu_row)
        all_kappa.append(kap_row)
        all_g.append(g_row)
    return per_seed, np.array(all_mu), np.array(all_kappa), np.array(all_g)


# ── Plot (publication-grade, matching make_viz_best.py style) ──────────
def plot_memtools_comparison(per_seed, all_mu, all_kappa, all_g,
                              raw_seed_means, save_path):
    human_mu = np.array(HUMAN_ERRORS)
    mu_mean = all_mu.mean(axis=0)
    mu_sd = all_mu.std(axis=0)
    kap_mean = all_kappa.mean(axis=0)
    kap_sd = all_kappa.std(axis=0)
    g_mean = all_g.mean(axis=0)
    g_sd = all_g.std(axis=0)

    raw_mean = raw_seed_means.mean(axis=0)
    raw_sd = raw_seed_means.std(axis=0)

    test_pct = np.array(TEST_POSITIONS) * 100

    # ── Figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 5.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 0.85, 1.05], wspace=0.34)

    # ── Panel A: Error curves (raw error, MemToolbox mu, human mu) ────
    ax = fig.add_subplot(grid[0])

    # Raw mean error — match TCM style from make_viz_best.py panel A
    ax.fill_between(test_pct, raw_mean - raw_sd, raw_mean + raw_sd,
                    color=MODEL_RED, alpha=0.2, linewidth=0)
    ax.plot(test_pct, raw_mean, "o-", color=MODEL_RED, linewidth=2.5,
            markersize=9, markeredgecolor="white", markeredgewidth=0.5,
            label=r"TCM raw error (mean $\pm$ SD)")

    # MemToolbox mu — blue squares for contrast
    ax.fill_between(test_pct, mu_mean - mu_sd, mu_mean + mu_sd,
                    color=MEMTOOLS_BLUE_LIGHT, alpha=0.22, linewidth=0)
    ax.plot(test_pct, mu_mean, "s-", color=MEMTOOLS_BLUE, linewidth=2.5,
            markersize=8, markeredgecolor="white", markeredgewidth=0.5,
            label=r"TCM MemToolbox $\mu$ (mean $\pm$ SD)")

    # Human mu
    ax.plot(test_pct, human_mu, "D-", color=HUMAN_BLACK,
            linewidth=2.5, markersize=9,
            label=r"Human $\mu$ (Hu et al.)", zorder=5)

    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("True position (%)")
    ax.set_ylabel(r"Bias $\mu$ (% of timeline)")
    ax.set_xlim(10, 90)
    ax.set_title(r"A  MemToolbox $\mu$ vs. Raw Error vs. Human",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, loc="upper right")

    # ── Panel B: mu parity ────────────────────────────────────────────
    ax = fig.add_subplot(grid[1])

    ax.plot([-15, 20], [-15, 20], "--", color="#CCCCCC", linewidth=1.2, zorder=0)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)

    # Raw error vs human — red to match panel A
    ax.scatter(human_mu, raw_mean, c=MODEL_RED, s=80, marker="o",
               edgecolors="white", linewidth=1, zorder=5,
               label="TCM raw error")
    # MemToolbox mu vs human
    ax.scatter(human_mu, mu_mean, c=MEMTOOLS_BLUE, s=80, marker="o",
               edgecolors="white", linewidth=1, zorder=6,
               label=r"MemToolbox $\mu$")

    # Annotate positions
    for i, pf in enumerate(TEST_POSITIONS):
        ax.annotate(f"{pf:.0%}", (human_mu[i] + 0.8, raw_mean[i] + 1.0),
                    fontsize=7.5, color=MODEL_RED, alpha=0.7)
        ax.annotate(f"{pf:.0%}", (human_mu[i] + 0.8, mu_mean[i] - 1.8),
                    fontsize=7.5, color=MEMTOOLS_BLUE, alpha=0.7)

    # R²
    ss_res_raw = np.sum((human_mu - raw_mean) ** 2)
    ss_tot = np.sum((human_mu - np.mean(human_mu)) ** 2)
    r2_raw = 1 - ss_res_raw / ss_tot
    ss_res_mu = np.sum((human_mu - mu_mean) ** 2)
    r2_mu = 1 - ss_res_mu / ss_tot

    ax.text(0.97, 0.08,
            f"Raw  $R^2$ = {r2_raw:.3f}\n"
            f"MemT $R^2$ = {r2_mu:.3f}",
            transform=ax.transAxes, fontsize=9, ha="right",
            va="bottom", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDDDDD", alpha=0.85))

    ax.set_xlabel(r"Human $\mu$ (% of timeline)")
    ax.set_ylabel("Model estimate (% of timeline)")
    ax.set_xlim(-12, 18)
    ax.set_ylim(-12, 18)
    ax.set_aspect("equal")
    ax.set_title(r"B  Model vs. Human $\mu$", fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")

    # ── Panel C: kappa and guess rate ─────────────────────────────────
    ax = fig.add_subplot(grid[2])
    ax_kappa = ax
    ax_guess = ax.twinx()

    # kappa bars (left axis)
    bars_k = ax_kappa.bar(test_pct - 2.2, kap_mean, 4.0,
                          color=KAPPA_GREEN_LIGHT, edgecolor=KAPPA_GREEN,
                          linewidth=0.8, yerr=kap_sd, capsize=3,
                          label=r"$\kappa$ (concentration)")
    # kappa individual seeds
    for seed_i in range(len(SEEDS)):
        ax_kappa.scatter(test_pct - 2.2 + np.random.uniform(-2, 2, 4),
                         all_kappa[seed_i], color=KAPPA_GREEN, s=18,
                         alpha=0.5, edgecolors="white", linewidth=0.4,
                         zorder=5)

    # guess rate bars (right axis, scaled to 0-100%)
    bars_g = ax_guess.bar(test_pct + 2.2, g_mean * 100, 4.0,
                          color=GUESS_ORANGE_LIGHT, edgecolor=GUESS_ORANGE,
                          linewidth=0.8, yerr=g_sd * 100, capsize=3,
                          label="$g$ (guess rate %)")
    for seed_i in range(len(SEEDS)):
        ax_guess.scatter(test_pct + 2.2 + np.random.uniform(-2, 2, 4),
                         all_g[seed_i] * 100, color=GUESS_ORANGE, s=18,
                         alpha=0.5, edgecolors="white", linewidth=0.4,
                         zorder=5)

    ax_kappa.set_xlabel("True position (%)")
    ax_kappa.set_ylabel(r"Concentration $\kappa$", color=KAPPA_GREEN)
    ax_kappa.tick_params(axis="y", labelcolor=KAPPA_GREEN)
    ax_kappa.set_xlim(5, 95)
    ax_kappa.set_ylim(0, np.ceil(kap_mean.max() + kap_sd.max() + 1))

    ax_guess.set_ylabel("Guess rate $g$ (%)", color=GUESS_ORANGE)
    ax_guess.tick_params(axis="y", labelcolor=GUESS_ORANGE)
    ax_guess.set_ylim(0, max(20, np.ceil((g_mean.max() + g_sd.max()) * 100) + 5))

    # Combined legend
    lines_k, labels_k = ax_kappa.get_legend_handles_labels()
    lines_g, labels_g = ax_guess.get_legend_handles_labels()
    ax_kappa.legend(lines_k + lines_g, labels_k + labels_g,
                    fontsize=7.5, loc="upper left")

    ax_kappa.set_title(r"C  Model Parameters ($\kappa$, $g$)",
                       fontweight="bold", loc="left")

    # ── Suptitle ──────────────────────────────────────────────────────
    asymmetry_model = (
        abs(mu_mean[0]) + abs(mu_mean[1])
        - abs(mu_mean[2]) - abs(mu_mean[3])
    ) / 2
    asymmetry_human = (
        abs(human_mu[0]) + abs(human_mu[1])
        - abs(human_mu[2]) - abs(human_mu[3])
    ) / 2
    fig.suptitle(
        "MemToolbox Mixture Model Fit \u2014 TCM trial-level outputs  |  "
        rf"$\mu$ asymmetry: model {asymmetry_model:+.1f}  /  "
        f"human {asymmetry_human:+.1f}  |  "
        rf"$\kappa$ range: {kap_mean.min():.1f}\u2013{kap_mean.max():.1f}  |  "
        f"5 seeds, 2000 trials/position",
        fontsize=13, fontweight="bold", y=0.995,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\nFigure saved to {save_path}")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MemToolbox-equivalent mixture model fitting")
    print(f"  rho={RHO}  sigma_m={SIGMA_M}  seeds={SEEDS}")
    print(f"  trials/pos = {N_TRIALS_PER_POS}")
    print("=" * 60)

    # Load pre-computed raw mean errors per seed
    raw_seed_means = []
    for seed in SEEDS:
        run_dir = param_dir(RHO, SIGMA_M, D_CONTEXT, seed, base=RESULTS_BASE)
        errors, _, _ = load_results(run_dir)
        raw_seed_means.append([np.mean(errors[pf]) for pf in TEST_POSITIONS])
    raw_seed_means_arr = np.array(raw_seed_means)

    # Sample trial-level outputs & fit
    all_outputs = []
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        encoder, decoder = load_model_from_checkpoint(seed)
        outputs = sample_trials(encoder, decoder)
        all_outputs.append(outputs)
        print(f"  Sampled {N_TRIALS_PER_POS} trials x 4 positions")

    print("\n" + "-" * 40)
    print("Fitting MemToolbox Mixture Model...")
    print("-" * 40)
    per_seed, all_mu, all_kappa, all_g = fit_memtools_model(all_outputs)

    # Summary
    mu_m = all_mu.mean(axis=0)
    mu_s = all_mu.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    print("\n" + "=" * 60)
    print("Summary (mean +/- SEM across seeds)")
    print("-" * 60)
    for i, pf in enumerate(TEST_POSITIONS):
        raw_m = raw_seed_means_arr[:, i].mean()
        print(f"  {pf:.0%}  raw={raw_m:+.2f}  memtools_mu={mu_m[i]:+.2f}+/-{mu_s[i]:.2f}  "
              f"human_mu={HUMAN_ERRORS[i]:+.1f}")

    # Plot
    save_path = os.path.join(os.path.dirname(__file__),
                             "figures", "memtools_comparison.png")
    plot_memtools_comparison(per_seed, all_mu, all_kappa, all_g,
                              raw_seed_means_arr, save_path)


if __name__ == "__main__":
    main()
