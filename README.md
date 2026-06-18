# TCM Temporal Position Model

A minimal neural network demonstrating that forward-asymmetric errors in temporal position judgment emerge from the interaction of autoregressive context drift with attention-based position decoding — without hand-specified asymmetry.

> **Reference:** Hu et al., "Temporal location biases in episodic time reconstruction and is tracked by the central gyrus"

---

## Context

When people watch a continuous event and later judge where individual moments occurred in time, they systematically overestimate positions of early moments and underestimate late ones — a central-tendency effect. Critically, this bias is **asymmetric**: early moments are distorted more than late ones (Hu et al., ~2025).

Classical Bayesian accounts predict symmetric regression toward the midpoint. The Temporal Context Model (TCM; Howard & Kahana, 2002) proposes that events are encoded within a gradually drifting internal context vector, c_t = ρ·c_{t−1} + (1−ρ)·f_t, such that temporal relationships are recovered through contextual similarity. Does this forward-recurrent architecture, when trained end-to-end on temporal position judgment, naturally produce the asymmetric human error pattern?

**This repo answers that question.** The model contains no explicit asymmetry parameter — only a forward-only recurrence (ρ < 1) and an attention-based position decoder. The asymmetry emerges from their interaction.

---

## Quick Start

```bash
pip install -r requirements.txt

# Full replication (trains all models, generates figures)
python sweep.py --rhos "0.70,0.80,0.90,0.95" --sigmas "0.05" \
    --seeds "42,43,44,45,46" --epochs 4000 --d 64
python make_viz.py

# Single model quick test
python main.py --rho 0.95 --sigma 0.05
```

---

## Reproducing All Results

### Step 1: Train models

```bash
python sweep.py --rhos "0.70,0.80,0.90,0.95" --sigmas "0.05" \
    --seeds "42,43,44,45,46" --epochs 4000 --d 64 --batch 32
```

This trains 20 models (4 ρ × 5 seeds) at the best-fit σ_m = 0.05. Results save to `logs/r{rho}_s0.05_d64/seed{seed}/` as `.npz` files. Training takes ~30–45 minutes on CPU.

To also sweep measurement noise levels:

```bash
python sweep.py --rhos "0.70,0.75,0.80,0.85,0.90,0.95" \
    --sigmas "0.05,0.10,0.15" --seeds "42,43,44,45,46" \
    --epochs 4000 --d 64 --batch 32
```

### Step 2: Generate the Summary figure

```bash
python make_viz.py
```

Produces `figures/tcm_viz_best_sigma.png` — a 2×3 panel figure showing:
- **A** — Signed error vs. true position for each ρ (mean ± SD across seeds)
- **B** — Asymmetry index vs. ρ (±SEM)
- **C** — Error distributions at the best-fit ρ
- **D** — Attention weights revealing the forward-skew mechanism
- **E** — Context similarity matrix, averaged across seeds
- **F** — Forward vs. backward similarity cross-section

### Step 3: Regenerate figures from saved data (no retraining)

```bash
python plot_from_data.py --all              # all individual-run figures
python plot_from_data.py --sweep-only       # sweep summary heatmap
```

---

## Architecture

**Encoder (TCM).** Items f_t are random vectors (proxy for perceptual frames). Context evolves autoregressively:

```
c_t = ρ · c_{t-1} + (1-ρ) · f_t,    c_0 = 0
```

ρ controls context persistence. At ρ = 0, c_t = f_t (no memory). At ρ ≈ 1, all contexts converge to a single vector (perfect memory, zero discriminability). ρ = 0.95 means each step retains 95% of the past.

**Decoder (Attention).** Position is not read out from c_q directly. Instead, the network compares c_q to all stored contexts c_1..c_L:

```
weights_i = softmax( Key(c_i)ᵀ · Query(c_q) )
position  = Σ w_i · (i / L)
```

The position template [0, 1/(L−1), ..., 1] provides each stored context's ground-truth relative position. The network's job is to learn *which* context to attend to, not *where* each position is.

**Training.** Variable-length sequences (60–180 items), MSE loss on relative position. Independent Gaussian noise (σ_m) is added to c_q at retrieval, introducing uncertainty that drives central-tendency effects.

**Why asymmetry emerges.** With ρ < 1, the context similarity is asymmetric: cos(c_q, c_{q+k}) > cos(c_q, c_{q−k}) for all k > 0. This is because c_{q+k} = ρᵏ·c_q + noise (forward, preserving c_q's direction), while c_q's composition of c_{q−k} is diluted by intervening items (backward). The softmax attention weights inherit this skew, shifting the position estimate forward — more so for early positions (which have more positions ahead of them to erroneously attend to).

**Critical control.** An MLP decoder that reads out position from c_q alone (no cross-context comparison) produces **zero asymmetry** — confirming the asymmetry requires the attention mechanism, not just the context encoding.

---

## Main Results

### Emergent asymmetry with σ_m = 0.05 (mean ± SD, 5 seeds per ρ)

| ρ | Error at 20% | Error at 40% | Error at 60% | Error at 80% | Asymmetry |
|:--|:--|:--|:--|:--|:--|
| 0.95 | +16.8 | +6.4 | −2.2 | −11.1 | **+5.0** |
| 0.90 | +10.8 | +4.4 | −1.8 | −8.0 | +2.7 |
| 0.80 | +7.3 | +3.1 | −1.3 | −5.5 | +1.8 |
| 0.70 | +4.8 | +2.0 | −0.7 | −3.3 | +1.4 |
| **Human** | **+13.4** | **+7.6** | **−1.9** | **−7.4** | **+5.9** |

Key observations:

- Forward asymmetry emerges at all ρ < 1, peaks at ρ = 0.95, and vanishes at ρ = 1.0 (inverted-U pattern). This is not trivially predicted — both extremes (ρ → 0, ρ → 1) produce zero asymmetry.
- The asymmetry magnitude at the best-fit ρ matches human data (+5.0 vs. +5.9).
- Error magnitudes increase with ρ because higher persistence makes contexts less discriminable → larger central tendency.

### Computational mechanism

The source of asymmetry is a forward-backward gap in context similarity. For a query at position 50 with ρ = 0.95:

- cos(c_50, c_51) ≈ 0.978 vs. cos(c_50, c_49) ≈ 0.959 (gap = +0.019)
- cos(c_50, c_52) ≈ 0.989 vs. cos(c_50, c_48) ≈ 0.931 (gap = +0.058)
- The gap grows to ~0.07 at k = 3–4

This small per-position gap accumulates across the softmax distribution, shifting the attention center-of-mass forward by several position units.

---

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                    # Train + eval single model
├── sweep.py                   # Parameter sweep (multi-seed)
├── plot_from_data.py          # Plot from saved logs (no retraining)
├── make_viz.py                # Publication-ready multi-panel figure
├── src/
│   ├── config.py              # All parameters & defaults
│   ├── model.py               # TCMEncoder + AttentionDecoder
│   ├── train.py               # Training & evaluation
│   ├── analysis.py            # Statistics & loss functions
│   ├── plotting.py            # Figure generation
│   └── utils.py               # I/O, paths, noise helpers
├── logs/                      # Model outputs (gitignored, auto-created)
│   └── r{rho}_s{sigma}_d{dim}/seed{seed}/
├── figures/                   # Generated figures (gitignored, auto-created)
└── legacy/                    # Development history (not imported)
```

---

## Parameters

| Parameter | CLI flag | Default | Description |
|:--|:--|:--|:--|
| ρ | `--rho` | 0.95 | Context persistence; 0 = no memory, 1 = perfect |
| σ_m | `--sigma` | 0.10 | Measurement noise SD added at retrieval |
| d | `--d` | 64 | Hidden/context dimension |
| epochs | `--epochs` | 4000 | Training epochs per model |
| seed | `--seed` | 42 | Random seed for reproducibility |
| seeds | `--seeds` | — | Comma-separated seed list for multi-seed sweeps |

---

## Critical Assumptions

1. **Attention-based position decoding.** The model assumes position is reconstructed by comparing the queried context against all stored contexts, rather than read out directly from a scalar time code. An MLP decoder (direct readout) produces zero asymmetry — supporting the necessity of cross-context comparison, but only within this model family.

2. **Perfect position template.** The decoder has access to the exact relative position (i/L) of every stored context. This is a simplification — human participants don't have a pre-indexed timeline. Future work should test whether asymmetry survives when the decoder must reconstruct position from context alone.

3. **Random item vectors.** Items are isotropic Gaussian vectors with no semantic structure. Adding realistic temporal autocorrelation or shot-boundary structure could change the quantitative results.

---
