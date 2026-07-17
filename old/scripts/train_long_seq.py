#!/usr/bin/env python3
"""Train TCM models with longer training sequences (150--180 steps).

The purpose is to test whether the forward-asymmetric position bias
replicates when the model is trained on longer, narrower-range sequences
compared with the default 60--180 regime.

Training: 5 seeds, ρ=0.95, σ_m=0.05, LR=5e-2, 4K epochs.
Output:  logs_longseq/
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analysis import compute_human_loss, compute_summary_stats
from src.config import D_ITEM, BATCH_SIZE as _CFG_BS
from src.model import TCMEncoder, AttentionDecoder
from src.train import make_items, evaluate_model, get_attention_weights
from src.utils import apply_noise, log_dir, save_results

# ── Experiment constants ──────────────────────────────────────────────
RHO = 0.95
SIGMA = 0.05
D_CONTEXT = 64
LR = 5e-2
EPOCHS = 4000
BATCH_SIZE = 32
SEEDS = [42, 43, 44, 45, 46]

SEQ_MIN = 150
SEQ_MAX = 180
LOGS_ROOT = os.path.join(os.path.dirname(__file__), "logs_longseq")


def train_one(rho, sigma_m, epochs, batch_size, d_item, d_context,
              seed, lr, seq_min, seq_max):
    """Train a single TCM model (duplicated from src.train to allow
    custom seq_min / seq_max at call time)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = TCMEncoder(rho, d_item, d_context, use_proj=True)
    decoder = AttentionDecoder(d_context, use_position_template=True)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()

    final_loss = None
    for epoch in range(epochs):
        L = np.random.randint(seq_min, seq_max + 1)
        items = make_items(batch_size, L, d_item)
        f, c = encoder(items)
        q = torch.randint(0, L, (batch_size,))
        c_q = apply_noise(c[torch.arange(batch_size), q, :], sigma_m)
        pred = decoder(c_q, c)
        loss = criterion(pred, q.float() / L)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optimizer.step()
        final_loss = loss.item()

    return encoder, decoder, final_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="retrain even when checkpoint exists")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                        help="comma-separated seed list")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    for run_idx, seed in enumerate(seeds, start=1):
        run_dir = log_dir(RHO, SIGMA, D_CONTEXT, seed, base=LOGS_ROOT)
        checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path) and not args.force:
            print(f"[{run_idx}/{len(seeds)}] seed={seed}: checkpoint exists ─ skip")
            continue

        print(f"[{run_idx}/{len(seeds)}] seed={seed}: training", flush=True)
        encoder, decoder, train_mse = train_one(
            RHO, SIGMA, epochs=EPOCHS, batch_size=BATCH_SIZE,
            d_item=D_ITEM, d_context=D_CONTEXT, seed=seed, lr=LR,
            seq_min=SEQ_MIN, seq_max=SEQ_MAX,
        )

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
                "seq_min": SEQ_MIN,
                "seq_max": SEQ_MAX,
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
            "seq_min": SEQ_MIN,
            "seq_max": SEQ_MAX,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        }, checkpoint_path)

        means_text = ", ".join(f"{m:+.2f}" for m in means)
        print(
            f"    errors=[{means_text}]  "
            f"asymmetry={asymmetry:+.2f}  "
            f"human_loss={human_loss:.2f}",
            flush=True,
        )

    print("All seeds trained. Run `python make_viz_long_seq.py` for the figure.")


if __name__ == "__main__":
    main()
