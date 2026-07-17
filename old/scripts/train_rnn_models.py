#!/usr/bin/env python3
"""Train the five-seed simple-RNN control shown in make_viz_best.py."""

import argparse
import os

import numpy as np
import torch

from src.analysis import compute_human_loss, compute_summary_stats
from src.config import D_ITEM
from src.model import AttentionDecoder, SimpleRNNEncoder
from src.train import evaluate_model, get_attention_weights, make_items
from src.utils import apply_noise, log_dir, save_results


SIGMA = 0.05
D_CONTEXT = 64
LR = 5e-4
EPOCHS = 4000
BATCH_SIZE = 32
SEEDS = [42, 43, 44, 45, 46]


def train_one(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = SimpleRNNEncoder(D_ITEM, D_CONTEXT)
    decoder = AttentionDecoder(D_CONTEXT, use_position_template=True)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LR)
    criterion = torch.nn.MSELoss()

    for _ in range(EPOCHS):
        length = np.random.randint(60, 181)
        items = make_items(BATCH_SIZE, length, D_ITEM)
        _, contexts = encoder(items)
        query_indices = torch.randint(0, length, (BATCH_SIZE,))
        queries = apply_noise(
            contexts[torch.arange(BATCH_SIZE), query_indices], SIGMA)
        predictions = decoder(queries, contexts)
        loss = criterion(predictions, query_indices.float() / length)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    return encoder, decoder, float(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()

    logs_root = os.path.join(os.path.dirname(__file__), "logs_rnn")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    for run_index, seed in enumerate(seeds, start=1):
        run_dir = log_dir(0.0, SIGMA, D_CONTEXT, seed, base=logs_root)
        errors_path = os.path.join(run_dir, "errors.npz")
        if os.path.exists(errors_path) and not args.force:
            print(f"[{run_index}/{len(seeds)}] seed={seed}: results exist")
            continue

        print(f"[{run_index}/{len(seeds)}] seed={seed}: training", flush=True)
        encoder, decoder, train_mse = train_one(seed)
        errors = evaluate_model(encoder, decoder, SIGMA, d_item=D_ITEM)
        attention = get_attention_weights(
            encoder, decoder, SIGMA, d_item=D_ITEM)
        means, _, asymmetry = compute_summary_stats(errors)
        human_loss = compute_human_loss(means)
        save_results(
            run_dir, errors, attention, rho=0.0, sigma_m=SIGMA,
            metadata={
                "model": "simple_rnn",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "d": D_CONTEXT,
                "seed": seed,
                "loss": human_loss,
                "asymmetry": asymmetry,
                "train_mse": train_mse,
            },
        )
        text = ", ".join(f"{mean:+.2f}" for mean in means)
        print(
            f"    errors=[{text}]  asymmetry={asymmetry:+.2f}  "
            f"human_loss={human_loss:.2f}", flush=True)


if __name__ == "__main__":
    main()
