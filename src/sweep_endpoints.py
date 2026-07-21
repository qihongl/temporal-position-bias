#!/usr/bin/env python3
"""
TCM Endpoint Sweep Runner — supports multiple decoder variants.

Usage:
  python sweep_endpoints.py --variant v1     # constrained attention
  python sweep_endpoints.py --variant v2     # endpoint-conditioned query
  python sweep_endpoints.py --variant raw    # raw scalar projection
  python sweep_endpoints.py --variant learned # learned MLP projection
"""

import argparse
import numpy as np
import os
import sys
import gc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import (
    D_ITEM, D_CONTEXT, SWEEP_EPOCHS, SWEEP_BATCH,
    TEST_POSITIONS, HUMAN_ERRORS, SEQ_MIN_ENDPOINT, SEQ_MAX,
    INTERVAL_LENGTHS_TEST,
)
from src.train_endpoints import (
    train_model_endpoints, train_model_endpoints_v2, train_model_projection,
    evaluate_model_endpoints, evaluate_model_endpoints_v2, evaluate_model_projection,
)
from src.analysis import compute_summary_stats, compute_human_loss
from src.utils import log_dir, save_results, ensure_dir, discover_runs


def save_interval_results(save_path, interval_results):
    ensure_dir(save_path)
    for ilen, ilen_results in interval_results.items():
        path = os.path.join(save_path, f'errors_i{ilen}.npz')
        np.savez(path,
                 p20=ilen_results[0.20], p40=ilen_results[0.40],
                 p60=ilen_results[0.60], p80=ilen_results[0.80])


VARIANTS = {
    'v1':  ('ConstrainedAttention', train_model_endpoints, evaluate_model_endpoints),
    'v2':  ('EndpointConditioned', train_model_endpoints_v2, evaluate_model_endpoints_v2),
    'raw': ('RawProjection', train_model_projection, evaluate_model_projection),
    'learned': ('LearnedProjection', None, None),
}


def main():
    parser = argparse.ArgumentParser(description='TCM Endpoint Sweep')
    parser.add_argument('--variant', type=str, default='v1',
                        choices=['v1', 'v2', 'raw', 'learned'],
                        help='Decoder variant')
    parser.add_argument('--rhos', type=str, default='0.70,0.80,0.90,0.95')
    parser.add_argument('--sigmas', type=str, default='0.05')
    parser.add_argument('--epochs', type=int, default=SWEEP_EPOCHS)
    parser.add_argument('--batch', type=int, default=SWEEP_BATCH)
    parser.add_argument('--d', type=int, default=D_CONTEXT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--query-type', type=str, default='context',
                        choices=['context', 'item'],
                        help='Query type: context (c_q) or item (f_q)')
    args = parser.parse_args()

    rhos = [float(r) for r in args.rhos.split(',')]
    sigmas = [float(s) for s in args.sigmas.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else [args.seed]
    base_dir = ROOT
    logs_root = os.path.join(base_dir, 'logs', f'endpoints_{args.variant}')
    if args.query_type == 'item':
        logs_root += '_itemq'

    name, train_fn, eval_fn = VARIANTS[args.variant]

    total = len(rhos) * len(sigmas) * len(seeds)
    print("=" * 60)
    print(f"TCM Endpoint Sweep ({args.variant}): {total} models")
    print(f"  ρ: {rhos}")
    print(f"  σ_m: {sigmas}")
    print(f"  seeds: {seeds}")
    print(f"  d: {args.d}, epochs: {args.epochs}")
    print(f"  Training seq: {SEQ_MIN_ENDPOINT}–{SEQ_MAX}")
    test_intervals = [16, 32, 64]
    print(f"  Test intervals: {test_intervals}")
    print(f"  Decoder: {name}")
    print("=" * 60)

    for i, rho in enumerate(rhos):
        for j, sm in enumerate(sigmas):
            for seed in seeds:
                counter = i * len(sigmas) * len(seeds) + j * len(seeds) + seeds.index(seed) + 1
                label = f"ρ={rho:.2f} σ_m={sm:.2f} seed={seed}"
                print(f"\n[{counter}/{total}] {label}")

                ldir = log_dir(rho, sm, args.d, seed, base=logs_root)
                if os.path.exists(os.path.join(ldir, 'errors_i16.npz')):
                    print(f"  → Already completed, skipping")
                    continue

                if args.variant == 'learned':
                    encoder, decoder, train_mse = train_model_projection(
                        rho, sm, epochs=args.epochs, batch_size=args.batch,
                        d_item=D_ITEM, d_context=args.d, seed=seed,
                        seq_min=SEQ_MIN_ENDPOINT, seq_max=SEQ_MAX,
                        learned=True,
                    )
                    interval_results = evaluate_model_projection(
                        encoder, decoder, sm, d_item=D_ITEM,
                        interval_lengths=test_intervals,
                    )
                elif args.variant == 'raw':
                    encoder, decoder, train_mse = train_model_projection(
                        rho, sm, epochs=args.epochs, batch_size=args.batch,
                        d_item=D_ITEM, d_context=args.d, seed=seed,
                        seq_min=SEQ_MIN_ENDPOINT, seq_max=SEQ_MAX,
                        learned=False,
                    )
                    interval_results = evaluate_model_projection(
                        encoder, decoder, sm, d_item=D_ITEM,
                        interval_lengths=test_intervals,
                    )
                else:
                    encoder, decoder, train_mse = train_fn(
                        rho, sm, epochs=args.epochs, batch_size=args.batch,
                        d_item=D_ITEM, d_context=args.d, seed=seed,
                        seq_min=SEQ_MIN_ENDPOINT, seq_max=SEQ_MAX,
                        query_type=args.query_type,
                    )
                    interval_results = eval_fn(
                        encoder, decoder, sm, d_item=D_ITEM,
                        interval_lengths=test_intervals,
                        query_type=args.query_type,
                    )

                print(f"  → results by interval length:")
                for ilen in test_intervals:
                    means = [np.mean(interval_results[ilen][pf]) for pf in TEST_POSITIONS]
                    asm = (abs(means[0])+abs(means[1])-abs(means[2])-abs(means[3]))/2
                    print(f"     i{ilen:>2}: [{means[0]:.1f},{means[1]:.1f},{means[2]:.1f},{means[3]:.1f}] asm={asm:+.1f}")

                all_agg = {pf: [] for pf in TEST_POSITIONS}
                for ilen in test_intervals:
                    for pf in TEST_POSITIONS:
                        all_agg[pf].extend(interval_results[ilen][pf].tolist())
                all_agg = {pf: np.array(v) for pf, v in all_agg.items()}
                agg_means, _, agg_asm = compute_summary_stats(all_agg)
                loss_val = compute_human_loss(agg_means)
                eval_mae = float(np.mean([np.mean(np.abs(all_agg[pf])) for pf in TEST_POSITIONS]))
                print(f"  → aggregate: [{agg_means[0]:.1f},{agg_means[1]:.1f},{agg_means[2]:.1f},{agg_means[3]:.1f}] asm={agg_asm:+.1f}, loss={loss_val:.1f}, MAE={eval_mae:.2f}%")

                save_interval_results(ldir, interval_results)
                save_results(ldir, all_agg, {}, rho, sm,
                             metadata={'epochs': args.epochs, 'loss': loss_val,
                                       'asymmetry': agg_asm, 'd': args.d, 'seed': seed,
                                       'train_mse': train_mse, 'eval_mae': eval_mae,
                                       'model_type': f'projection_{args.variant}'})
                print(f"  → logs: {ldir}")

                del encoder, decoder; gc.collect()

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"  Human: {HUMAN_ERRORS}")
    runs = discover_runs(logs_root)
    print(f"  Total completed runs: {len(runs)}")


if __name__ == '__main__':
    main()
