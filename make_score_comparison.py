#!/usr/bin/env python3
"""Compare raw similarity, raw dot products, and learned Q/K scores.

The first seven panels use the scores at their native scales. The final panel
uses paired, scale-matched counterfactuals: each alternative score field is
multiplied by one seed-level scalar so that its centered-logit SD matches the
full model. This isolates score geometry from the trivial effect of making the
softmax uniformly flat.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from make_viz_best import (
    D_ITEM, FIGURE_PATH, HUMAN_ERRORS, SEEDS, SEQ_TEST, TEST_POSITIONS,
    best_run_paths, load_checkpoint,
)


OUTPUT_PATH = os.path.join(
    os.path.dirname(FIGURE_PATH), "tcm_score_comparison.png")
STAGES = ["raw_cosine", "raw_dot", "learned"]
COUNTERFACTUALS = [
    "full", "raw_cosine", "raw_dot", "projected_cosine", "flat_key_norm",
]
COLORS = {
    "full": "#111111",
    "raw_cosine": "#7570B3",
    "raw_dot": "#1B9E77",
    "projected_cosine": "#D95F02",
    "flat_key_norm": "#E7298A",
}
LABELS = {
    "full": "Full learned Q/K",
    "raw_cosine": "Raw cosine",
    "raw_dot": "Raw dot (K=Q=I)",
    "projected_cosine": "Projected cosine",
    "flat_key_norm": "Flattened key norms",
}


def center_rows(logits):
    return logits - logits.mean(dim=-1, keepdim=True)


def scale_match(reference, alternative):
    """Match centered-logit SD with one scalar, preserving score geometry."""
    reference_centered = center_rows(reference)
    alternative_centered = center_rows(alternative)
    denominator = alternative_centered.std().clamp_min(1e-8)
    factor = reference_centered.std() / denominator
    return alternative_centered * factor, float(factor.item())


def error_curve(logits):
    weights = F.softmax(logits, dim=-1)
    template = torch.linspace(0, 1, SEQ_TEST)
    targets = torch.arange(SEQ_TEST).float() / SEQ_TEST
    predictions = (weights * template.view(1, 1, -1)).sum(dim=-1)
    return (100 * (predictions - targets.view(1, -1))).mean(dim=0)


def diagnose_one(encoder, decoder, n_sequences, batch_size, eval_seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(eval_seed)
    collected = {
        "raw_context_cosine": [],
        "raw_cosine_logits": [],
        "raw_dot_logits": [],
        "projected_cosine_logits": [],
        "flat_key_logits": [],
        "learned_logits": [],
    }

    seen = 0
    with torch.no_grad():
        while seen < n_sequences:
            current_batch = min(batch_size, n_sequences - seen)
            items = torch.randn(
                current_batch, SEQ_TEST, D_ITEM, generator=generator)
            _, contexts = encoder(items)
            noise = 0.05 * torch.randn(
                contexts.shape, generator=generator,
                dtype=contexts.dtype, device=contexts.device)
            noisy_queries = F.normalize(contexts + noise, dim=-1)
            normalized_contexts = F.normalize(contexts, dim=-1)

            keys = decoder.key(contexts)
            queries = decoder.query(noisy_queries)
            learned_logits = torch.bmm(
                queries, keys.transpose(1, 2)) / decoder.temperature

            raw_context_cosine = torch.bmm(
                normalized_contexts, normalized_contexts.transpose(1, 2))
            raw_cosine_logits = torch.bmm(
                noisy_queries, normalized_contexts.transpose(1, 2))
            raw_dot_logits = torch.bmm(
                noisy_queries, contexts.transpose(1, 2)) / decoder.temperature
            projected_cosine_logits = torch.bmm(
                F.normalize(queries, dim=-1),
                F.normalize(keys, dim=-1).transpose(1, 2))

            key_norms = keys.norm(dim=-1, keepdim=True)
            mean_key_norm = key_norms.mean(dim=1, keepdim=True)
            flat_keys = F.normalize(keys, dim=-1) * mean_key_norm
            flat_key_logits = torch.bmm(
                queries, flat_keys.transpose(1, 2)) / decoder.temperature

            for name, values in (
                ("raw_context_cosine", raw_context_cosine),
                ("raw_cosine_logits", raw_cosine_logits),
                ("raw_dot_logits", raw_dot_logits),
                ("projected_cosine_logits", projected_cosine_logits),
                ("flat_key_logits", flat_key_logits),
                ("learned_logits", learned_logits),
            ):
                collected[name].append(values.cpu())
            seen += current_batch

    values = {name: torch.cat(parts, dim=0)
              for name, parts in collected.items()}
    learned = values["learned_logits"]

    raw_cosine_matched, raw_cosine_factor = scale_match(
        learned, values["raw_cosine_logits"])
    raw_dot_matched, raw_dot_factor = scale_match(
        learned, values["raw_dot_logits"])
    projected_cosine_matched, projected_cosine_factor = scale_match(
        learned, values["projected_cosine_logits"])
    flat_key_matched, flat_key_factor = scale_match(
        learned, values["flat_key_logits"])

    return {
        "matrices": {
            "raw_cosine": values["raw_context_cosine"].mean(dim=0).numpy(),
            "raw_dot": center_rows(values["raw_dot_logits"]).mean(dim=0).numpy(),
            "learned": center_rows(learned).mean(dim=0).numpy(),
        },
        "errors": {
            "full": error_curve(learned).numpy(),
            "raw_cosine": error_curve(raw_cosine_matched).numpy(),
            "raw_dot": error_curve(raw_dot_matched).numpy(),
            "projected_cosine": error_curve(projected_cosine_matched).numpy(),
            "flat_key_norm": error_curve(flat_key_matched).numpy(),
        },
        "scale_factors": {
            "raw_cosine": raw_cosine_factor,
            "raw_dot": raw_dot_factor,
            "projected_cosine": projected_cosine_factor,
            "flat_key_norm": flat_key_factor,
        },
    }


def matched_lag_differences(matrix, max_lag=20):
    """Return query × lag values S[i,i+k] - S[i,i-k]."""
    differences = np.full((SEQ_TEST, max_lag), np.nan)
    for query in range(SEQ_TEST):
        for lag in range(1, max_lag + 1):
            if query - lag >= 0 and query + lag < SEQ_TEST:
                differences[query, lag - 1] = (
                    matrix[query, query + lag] - matrix[query, query - lag])
    return differences


def local_direction_index(matrix, max_lag=10):
    differences = matched_lag_differences(matrix, max_lag=max_lag)
    row_scale = np.std(matrix, axis=1)
    result = np.full(SEQ_TEST, np.nan)
    valid_rows = np.arange(max_lag, SEQ_TEST - max_lag)
    result[valid_rows] = (
        np.mean(differences[valid_rows], axis=1)
        / np.maximum(row_scale[valid_rows], 1e-8)
    )
    return result


def asymmetry(error_curve_values):
    selected = [error_curve_values[int(pf * SEQ_TEST)] for pf in TEST_POSITIONS]
    return (abs(selected[0]) + abs(selected[1])
            - abs(selected[2]) - abs(selected[3])) / 2


def matrix_panel(ax, matrix, title, raw_cosine=False):
    if raw_cosine:
        image = ax.imshow(matrix, origin="lower", aspect="equal", cmap="YlGnBu",
                          vmin=0, vmax=1, extent=(0, 100, 0, 100))
    else:
        limit = max(np.percentile(np.abs(matrix), 98), 1e-8)
        image = ax.imshow(matrix, origin="lower", aspect="equal", cmap="RdBu_r",
                          vmin=-limit, vmax=limit, extent=(0, 100, 0, 100))
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.55)
    ax.set(xlabel="Candidate position j (%)", ylabel="Query position i (%)")
    ax.set_title(title, fontweight="bold", loc="left")
    plt.colorbar(image, ax=ax, shrink=0.76)


def delta_panel(ax, differences, title):
    finite = differences[np.isfinite(differences)]
    limit = max(np.percentile(np.abs(finite), 98), 1e-8)
    image = ax.imshow(
        differences, origin="lower", aspect="auto", cmap="PiYG",
        vmin=-limit, vmax=limit, extent=(0.5, 20.5, 0, 100))
    ax.axhline(20, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axhline(40, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axhline(60, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axhline(80, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    rms = np.sqrt(np.mean(finite ** 2))
    ax.set(xlabel="Matched lag |k|", ylabel="Query position i (%)")
    ax.set_title(f"{title}\nRMS Δ={rms:.3g}", fontweight="bold", loc="left")
    plt.colorbar(image, ax=ax, shrink=0.76, label="forward − backward")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-sequences", type=int, default=200)
    parser.add_argument("--diagnostic-batch", type=int, default=50)
    args = parser.parse_args()

    diagnostics = []
    for seed, run_path in zip(SEEDS, best_run_paths()):
        print(f"Comparing score structures for seed={seed}...", flush=True)
        encoder, decoder = load_checkpoint(run_path)
        diagnostics.append(diagnose_one(
            encoder, decoder, args.diagnostic_sequences,
            args.diagnostic_batch, eval_seed=20260623))

    matrices_by_stage = {
        stage: np.stack([d["matrices"][stage] for d in diagnostics])
        for stage in STAGES
    }
    errors_by_condition = {
        condition: np.stack([d["errors"][condition] for d in diagnostics])
        for condition in COUNTERFACTUALS
    }

    mean_matrices = {
        stage: matrices_by_stage[stage].mean(axis=0) for stage in STAGES
    }
    mean_differences = {}
    for stage in STAGES:
        seed_differences = np.stack([
            matched_lag_differences(matrix)
            for matrix in matrices_by_stage[stage]
        ])
        valid_counts = np.sum(np.isfinite(seed_differences), axis=0)
        summed = np.nansum(seed_differences, axis=0)
        mean_differences[stage] = np.divide(
            summed, valid_counts,
            out=np.full_like(summed, np.nan), where=valid_counts > 0)

    fig = plt.figure(figsize=(21, 10.5))
    grid = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.33)

    matrix_panel(fig.add_subplot(grid[0, 0]), mean_matrices["raw_cosine"],
                 "A  Raw Context Cosine", raw_cosine=True)
    matrix_panel(fig.add_subplot(grid[0, 1]), mean_matrices["raw_dot"],
                 "B  Raw Dot Logits (K=Q=I)")
    matrix_panel(fig.add_subplot(grid[0, 2]), mean_matrices["learned"],
                 "C  Learned Q/K Logits")

    ax = fig.add_subplot(grid[0, 3])
    x = np.arange(SEQ_TEST) / SEQ_TEST * 100
    stage_colors = {
        "raw_cosine": "#7570B3", "raw_dot": "#1B9E77", "learned": "#D95F02"
    }
    for stage in STAGES:
        indices = np.stack([
            local_direction_index(matrix)
            for matrix in matrices_by_stage[stage]
        ])
        valid = np.all(np.isfinite(indices), axis=0)
        mean = np.full(SEQ_TEST, np.nan)
        sd = np.full(SEQ_TEST, np.nan)
        mean[valid] = np.mean(indices[:, valid], axis=0)
        sd[valid] = np.std(indices[:, valid], axis=0)
        ax.fill_between(x[valid], (mean - sd)[valid], (mean + sd)[valid],
                        color=stage_colors[stage], alpha=0.14, linewidth=0)
        ax.plot(x[valid], mean[valid], color=stage_colors[stage], linewidth=1.8,
                label=stage.replace("_", " "))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(xlabel="Query position (%)",
           ylabel="Mean matched-lag Δ / row SD", xlim=(0, 99))
    ax.set_title("D  Normalized Local Directionality\n(mean lags 1–10)",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8)

    delta_panel(fig.add_subplot(grid[1, 0]), mean_differences["raw_cosine"],
                "E  Raw-Cosine Matched-Lag Δ")
    delta_panel(fig.add_subplot(grid[1, 1]), mean_differences["raw_dot"],
                "F  Raw-Dot Matched-Lag Δ")
    delta_panel(fig.add_subplot(grid[1, 2]), mean_differences["learned"],
                "G  Learned-Logit Matched-Lag Δ")

    ax = fig.add_subplot(grid[1, 3])
    for condition in COUNTERFACTUALS:
        curves = errors_by_condition[condition]
        mean_curve = curves.mean(axis=0)
        asm = asymmetry(mean_curve)
        ax.plot(x, mean_curve, color=COLORS[condition], linewidth=2,
                label=f"{LABELS[condition]} (asm={asm:+.1f})")
    human_x = np.asarray(TEST_POSITIONS) * 100
    ax.plot(human_x, HUMAN_ERRORS, "D", color="#222222", markersize=6,
            label="Human")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(xlabel="Query position (%)", ylabel="Signed error (%)", xlim=(0, 99))
    ax.set_title("H  Scale-Matched Score Counterfactuals",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=6.8, loc="upper right")

    factors = {
        condition: np.mean([d["scale_factors"][condition]
                            for d in diagnostics])
        for condition in COUNTERFACTUALS if condition != "full"
    }
    factor_text = ", ".join(f"{name.replace('_', ' ')} ×{value:.1f}"
                            for name, value in factors.items())
    print("\nScale-matched counterfactual summary:")
    for condition in COUNTERFACTUALS:
        mean_curve = errors_by_condition[condition].mean(axis=0)
        print(f"  {LABELS[condition]:24s} asymmetry={asymmetry(mean_curve):+5.2f}")
    print("  Scale factors:", factor_text)
    fig.suptitle(
        "What do learned Q/K projections change?  "
        "Native score geometry (A–G) and paired scale-matched readouts (H)\n"
        f"Counterfactual scale factors: {factor_text}",
        fontsize=13, fontweight="bold", y=1.01)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
