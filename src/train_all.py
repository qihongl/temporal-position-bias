#!/usr/bin/env python3
"""
Re-train all TCM models with the fixed target (q/(L-1)).
Run all rhos × 20 seeds + RNN control × 5 seeds.
Saves to logs/fixed_target/ and logs/rnn_fixed/.
"""
import os, sys, gc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from src.config import D_ITEM, D_CONTEXT, SEQ_TEST, TEST_POSITIONS, HUMAN_ERRORS
from src.train import train_model, evaluate_model, train_rnn_model
from src.analysis import compute_summary_stats, compute_human_loss
from src.utils import log_dir, save_results

RHOS = [0.60, 0.70, 0.80, 0.90, 0.95]
SIGMA_M = 0.05
N_SEEDS = 20
NN_SEEDS = 5
D = D_CONTEXT

BASE_TCM = os.path.join(ROOT, 'logs', 'fixed_target')
BASE_RNN = os.path.join(ROOT, 'logs', 'rnn_fixed')

total = len(RHOS) * N_SEEDS + NN_SEEDS
counter = 0


def train_one(rho, sigma, seed, log_base):
    """Train one model, return means/asym/loss or None if already done."""
    ldir = log_dir(rho, sigma, D, seed, base=log_base)
    if os.path.exists(os.path.join(ldir, 'errors.npz')):
        return None  # skip

    encoder, decoder, train_mse = train_model(
        rho, sigma, seed=seed, d_item=D_ITEM, d_context=D)
    errors = evaluate_model(encoder, decoder, sigma, d_item=D_ITEM)
    means, _, asymmetry = compute_summary_stats(errors)
    loss_val = compute_human_loss(means)
    eval_mae = float(np.mean([np.mean(np.abs(errors[p])) for p in TEST_POSITIONS]))

    save_results(ldir, errors, {}, rho, sigma,
                 metadata={'epochs': 4000, 'loss': loss_val,
                           'asymmetry': asymmetry, 'd': D, 'seed': seed,
                           'train_mse': train_mse, 'eval_mae': eval_mae,
                           'fix': 'target_q_over_L_minus_1'})
    return means, asymmetry, loss_val, train_mse


def train_one_rnn(sigma, seed, log_base):
    """Train one RNN model."""
    ldir = log_dir(0.0, sigma, D, seed, base=log_base)
    if os.path.exists(os.path.join(ldir, 'errors.npz')):
        return None

    encoder, decoder, train_mse = train_rnn_model(
        sigma, seed=seed, d_item=D_ITEM, d_context=D)
    errors = evaluate_model(encoder, decoder, sigma, d_item=D_ITEM)
    means, _, asymmetry = compute_summary_stats(errors)
    loss_val = compute_human_loss(means)
    eval_mae = float(np.mean([np.mean(np.abs(errors[p])) for p in TEST_POSITIONS]))

    save_results(ldir, errors, {}, 0.0, sigma,
                 metadata={'epochs': 4000, 'loss': loss_val,
                           'asymmetry': asymmetry, 'd': D, 'seed': seed,
                           'train_mse': train_mse, 'eval_mae': eval_mae,
                           'model_type': 'rnn', 'fix': 'target_q_over_L_minus_1'})
    return means, asymmetry, loss_val, train_mse


print("=" * 60)
print(f"FULL SWEEP — Fixed target q/(L-1)")
print(f"  TCM: {len(RHOS)} rhos × {N_SEEDS} seeds = {len(RHOS)*N_SEEDS} models")
print(f"  RNN: {NN_SEEDS} seeds")
print(f"  Total: {total} models")
print(f"  σ_m = {SIGMA_M}")
print("=" * 60)

# ——— TCM models ———
for rho in RHOS:
    for seed in range(42, 42 + N_SEEDS):
        counter += 1
        label = f"TCM ρ={rho:.2f} seed={seed}"
        print(f"\n[{counter}/{total}] {label}", flush=True)

        result = train_one(rho, SIGMA_M, seed, BASE_TCM)
        if result is None:
            print(f"  → Already completed, skipping")
            continue
        means, asymmetry, loss_val, train_mse = result
        print(f"  means: [{means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f}, {means[3]:.1f}]")
        print(f"  asym: {asymmetry:+.1f}, loss: {loss_val:.1f}, MSE: {train_mse:.6f}")
        gc.collect()

# ——— RNN models ———
for seed in range(42, 42 + NN_SEEDS):
    counter += 1
    label = f"RNN seed={seed}"
    print(f"\n[{counter}/{total}] {label}", flush=True)

    result = train_one_rnn(SIGMA_M, seed, BASE_RNN)
    if result is None:
        print(f"  → Already completed, skipping")
        continue
    means, asymmetry, loss_val, train_mse = result
    print(f"  means: [{means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f}, {means[3]:.1f}]")
    print(f"  asym: {asymmetry:+.1f}, loss: {loss_val:.1f}, MSE: {train_mse:.6f}")
    gc.collect()

print("\n\nALL DONE.")
print(f"  TCM logs: {BASE_TCM}")
print(f"  RNN logs: {BASE_RNN}")
