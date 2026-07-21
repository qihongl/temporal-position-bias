#!/usr/bin/env python3
"""
Train TCM models with f_q (item) query instead of c_q (context) query.
All rhos, 5 seeds each. Also trains RNN with f_q query.
"""
import os, sys, gc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from src.config import D_ITEM, D_CONTEXT, SEQ_TEST, TEST_POSITIONS, HUMAN_ERRORS
from src.train import train_model, evaluate_model, train_rnn_model
from src.analysis import compute_summary_stats, compute_human_loss
from src.utils import log_dir, save_results

RHOS = [0.60, 0.70, 0.80, 0.90, 0.95]
SIGMA_M = 0.05
N_SEEDS = 5
D = D_CONTEXT

BASE_TCM = os.path.join(ROOT, 'logs', 'fq_query')
BASE_RNN = os.path.join(ROOT, 'logs', 'rnn_fq')

total = len(RHOS) * N_SEEDS + N_SEEDS  # + RNN
counter = 0

print("=" * 60)
print(f"F_Q QUERY SWEEP — query = f_q (projected item)")
print(f"  TCM: {len(RHOS)} rhos × {N_SEEDS} seeds = {len(RHOS)*N_SEEDS} models")
print(f"  RNN: {N_SEEDS} seeds")
print(f"  Total: {total} models")
print(f"  σ_m = {SIGMA_M}")
print("=" * 60)

for rho in RHOS:
    for seed in range(42, 42 + N_SEEDS):
        counter += 1
        label = f"TCM f_q ρ={rho:.2f} seed={seed}"
        print(f"\n[{counter}/{total}] {label}", flush=True)

        ldir = log_dir(rho, SIGMA_M, D, seed, base=BASE_TCM)
        if os.path.exists(os.path.join(ldir, 'errors.npz')):
            print(f"  → Already completed, skipping")
            continue

        encoder, decoder, train_mse = train_model(
            rho, SIGMA_M, seed=seed, d_item=D_ITEM, d_context=D,
            query_type='item')
        errors = evaluate_model(encoder, decoder, SIGMA_M, d_item=D_ITEM,
                                query_type='item')
        means, _, asymmetry = compute_summary_stats(errors)
        loss_val = compute_human_loss(means)
        eval_mae = float(np.mean([np.mean(np.abs(errors[p])) for p in TEST_POSITIONS]))

        print(f"  means: [{means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f}, {means[3]:.1f}]")
        print(f"  asym: {asymmetry:+.1f}, loss: {loss_val:.1f}, MSE: {train_mse:.6f}")

        save_results(ldir, errors, {}, rho, SIGMA_M,
                     metadata={'epochs': 4000, 'loss': loss_val,
                               'asymmetry': asymmetry, 'd': D, 'seed': seed,
                               'train_mse': train_mse, 'eval_mae': eval_mae,
                               'query_type': 'item', 'fix': 'target_q_over_L_minus_1'})
        print(f"  saved: {ldir}")
        gc.collect()

for seed in range(42, 42 + N_SEEDS):
    counter += 1
    label = f"RNN f_q seed={seed}"
    print(f"\n[{counter}/{total}] {label}", flush=True)

    ldir = log_dir(0.0, SIGMA_M, D, seed, base=BASE_RNN)
    if os.path.exists(os.path.join(ldir, 'errors.npz')):
        print(f"  → Already completed, skipping")
        continue

    encoder, decoder, train_mse = train_rnn_model(
        SIGMA_M, seed=seed, d_item=D_ITEM, d_context=D, query_type='item')
    errors = evaluate_model(encoder, decoder, SIGMA_M, d_item=D_ITEM,
                            query_type='item')
    means, _, asymmetry = compute_summary_stats(errors)
    loss_val = compute_human_loss(means)
    eval_mae = float(np.mean([np.mean(np.abs(errors[p])) for p in TEST_POSITIONS]))

    print(f"  means: [{means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f}, {means[3]:.1f}]")
    print(f"  asym: {asymmetry:+.1f}, loss: {loss_val:.1f}, MSE: {train_mse:.6f}")

    save_results(ldir, errors, {}, 0.0, SIGMA_M,
                 metadata={'epochs': 4000, 'loss': loss_val,
                           'asymmetry': asymmetry, 'd': D, 'seed': seed,
                           'train_mse': train_mse, 'eval_mae': eval_mae,
                           'model_type': 'rnn', 'query_type': 'item',
                           'fix': 'target_q_over_L_minus_1'})
    print(f"  saved: {ldir}")
    gc.collect()

print("\n\nALL DONE.")
print(f"  TCM logs: {BASE_TCM}")
print(f"  RNN logs: {BASE_RNN}")
