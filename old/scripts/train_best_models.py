#!/usr/bin/env python3
"""Train and checkpoint the five models used by make_viz_best.py.

The original ``logs_best`` artifacts contain behavioral summaries but not the
learned P/K/Q parameters.  Mechanism plots require those parameters, so this
script recreates the exact best configuration and stores a checkpoint beside
each run's existing ``.npz`` files.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analysis import compute_human_loss, compute_summary_stats
from src.config import D_ITEM
from src.train import evaluate_model, get_attention_weights, train_model
from src.utils import log_dir, save_results


RHO = 0.95
SIGMA = 0.05
D_CONTEXT = 64
LR = 5e-2
EPOCHS = 4000
BATCH_SIZE = 32
SEEDS = [42, 43, 44, 45, 46]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="retrain even when a checkpoint already exists")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                        help="comma-separated seed list")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    logs_root = os.path.join(os.path.dirname(__file__), "logs_best")

    for run_idx, seed in enumerate(seeds, start=1):
        run_dir = log_dir(RHO, SIGMA, D_CONTEXT, seed, base=logs_root)
        checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path) and not args.force:
            print(f"[{run_idx}/{len(seeds)}] seed={seed}: checkpoint exists")
            continue

        print(f"[{run_idx}/{len(seeds)}] seed={seed}: training", flush=True)
        encoder, decoder, train_mse = train_model(
            RHO, SIGMA, epochs=EPOCHS, batch_size=BATCH_SIZE,
            d_item=D_ITEM, d_context=D_CONTEXT, seed=seed,
            use_position_template=True, use_proj=True, lr=LR,
        )

        # Preserve the evaluation order used by the main sweep workflow.
        errors = evaluate_model(encoder, decoder, SIGMA, d_item=D_ITEM)
        attention = get_attention_weights(
            encoder, decoder, SIGMA, d_item=D_ITEM)
        means, _, asymmetry = compute_summary_stats(errors)
        human_loss = compute_human_loss(means)

        save_results(
            run_dir, errors, attention, RHO, SIGMA,
            metadata={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "d": D_CONTEXT,
                "seed": seed,
                "loss": human_loss,
                "asymmetry": asymmetry,
                "train_mse": train_mse,
                "use_position_template": True,
                "use_proj": True,
            },
        )
        torch.save({
            "rho": RHO,
            "sigma_m": SIGMA,
            "d_item": D_ITEM,
            "d_context": D_CONTEXT,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": seed,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        }, checkpoint_path)

        means_text = ", ".join(f"{m:+.2f}" for m in means)
        print(
            f"    errors=[{means_text}]  asymmetry={asymmetry:+.2f}  "
            f"human_loss={human_loss:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
