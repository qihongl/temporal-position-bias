# TCM Temporal Position Model

A minimal neural network demonstrating that forward-asymmetric errors in temporal position judgment emerge from the interaction of autoregressive context drift with an item-query-based position decoder — without hand-specified asymmetry.

> **Reference:** Hu et al., "Temporal location biases in episodic time reconstruction and is tracked by the central gyrus"

---

## Context

When people watch a continuous event and later judge where individual moments occurred in time, they systematically overestimate positions of early moments and underestimate late ones — a central-tendency effect. Critically, this bias is **asymmetric**: early moments are distorted more than late ones (Hu et al., ~2025).

Classical Bayesian accounts predict symmetric regression toward the midpoint. The Temporal Context Model (TCM; Howard & Kahana, 2002) proposes that events are encoded within a gradually drifting internal context vector, `c_t = ρ·c_{t−1} + (1−ρ)·f_t`, such that temporal relationships are recovered through contextual similarity.

**Can this forward-recurrent architecture, when given an item-level query and endpoint-constrained retrieval, produce the asymmetric human error pattern?**

---

## Model

### Study Phase (Encoding)

```
x_t ~ N(0, I)                              random items (proxy for movie frames)
f_t = normalize(P · x_t)                    P: [64×64] learnable projection
c_t = ρ · c_{t-1} + (1−ρ) · f_t            c_0 = 0

C = [c_1, ..., c_L]   stored in memory
F = [f_1, ..., f_L]   item representations also stored
```

### Recall Phase (Position Estimation)

Given interval bounds A, B and target index q ∈ (A, B):

```
f̃_q = normalize( f_q + ε )                  ε ~ N(0, σ_m²·I)   memory noise
q_vec = Q( f̃_q )                             Q: [64×64] learnable
K = K( C )                                   K: [64×64] learnable

scores_i = (K_i · q_vec) / T                 T = softplus(t_raw), learnable
           for i ∈ [A, B], masked otherwise

α = softmax(scores)
poŝ = Σ_i α_i · (i−A)/(B−A)                 relative position ∈ [0,1]

loss = MSE(poŝ, (q−A)/(B−A))
```

### Why Asymmetry Emerges

The key innovation is using the **raw item** `f_q` as the retrieval cue — not the accumulated context `c_q`. This creates a **unit-step** similarity profile:

```
c_i · f_q  ≈  0                              for i < q   (f_q not in c_i)
           ≈  (1−ρ)ρ^{i−q}                   for i ≥ q   (decaying signal)
```

All attention mass pushes **forward** of q. The forward shift is larger at early positions (more room ahead) than late positions (little room ahead):

```
Early q (20%):  many positions ahead  →  large forward shift →  large overestimation
Late q  (80%):  few positions ahead   →  small forward shift →  small underestimation

asymmetry = (|error_20%| + |error_40%| − |error_60%| − |error_80%|) / 2  >  0
```

The interval length modulates the bias: shorter intervals compress the forward shift into a smaller range, amplifying asymmetry.

### Learnable vs Fixed

| Learnable (gradients) | Fixed (hyperparameters / operations) |
|---|---|
| P [64×64] | ρ = 0.95 |
| Q [64×64] | σ_m = 0.05 |
| K [64×64] | TCM recurrence |
| t_raw (scalar) | L2 normalization |
| | softmax |
| Total: ~12,353 params | attention mask [A, B] |
| | position template (i−A)/(B−A) |

The asymmetry is NOT a learned parameter. It emerges from the structural interaction between the TCM's forward-auto-regression and the item query's unit-step similarity profile.

---

## Quick Start

```bash
pip install -r requirements.txt

# Train endpoint-constrained models with item query (20 seeds, ~1.5 hr CPU)
python sweep_endpoints.py --variant v1 --query-type item \
    --rhos "0.70,0.80,0.90,0.95" --sigmas "0.05" \
    --seeds "42,...,61" --epochs 4000 --d 64 --batch 32

# Generate per-interval visualization
python make_viz_endpoints_intervals.py
```

---

## Main Results

### Item query + constrained attention (20 seeds, σ_m = 0.05, ρ = 0.95)

| Interval | Error 20% | Error 40% | Error 60% | Error 80% | Asymmetry |
|:---------|:----------|:----------|:----------|:----------|:----------|
| i16 | +31.1 | +14.1 | −4.2 | −24.0 | **+8.5** |
| i32 | +29.8 | +12.5 | −5.3 | −25.2 | **+5.9** |
| i64 | +28.6 | +11.3 | −6.8 | −26.4 | +3.4 |
| **Human** | **+13.4** | **+7.6** | **−1.9** | **−7.4** | **+5.9** |

Key findings:
- Asymmetry **decreases with interval length** — longer intervals spread the forward decay tail, diluting the shift
- ρ = 0.95 at i32 **exactly matches human asymmetry** (+5.9 vs. +5.9)
- Error magnitudes are ~2× human, but the asymmetry magnitude matches
- Results are robust: seed-to-seed SD across 20 independent initializations is < 0.3 percentage points

### Effect of measurement noise (σ_m)

| σ_m | ρ=0.95, i32 asym | ρ=0.90, i32 asym | ρ=0.80, i32 asym |
|-----|-------------------|-------------------|-------------------|
| 0.05 | +5.9 | +9.0 | +4.2 |
| 0.10 | +4.6 | +7.8 | +5.3 |
| 0.20 | +2.4 | +4.0 | +4.1 |

Higher noise reduces asymmetry — the unit-step signal gets noisier, blurring the forward-bias gradient. At σ_m = 0.20 the bias is still positive but diminished.

---

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── config.py              # Parameters & defaults
│   ├── model.py               # TCMEncoder + decoder variants
│   ├── train.py               # Training & evaluation functions
│   ├── analysis.py            # Statistics & loss functions
│   ├── plotting.py            # Figure generation
│   └── utils.py               # I/O, paths, noise helpers
├── sweep_endpoints.py         # Endpoint-constrained sweep runner
├── train_endpoints.py         # Single-model endpoint trainer
├── sweep.py                   # Original (global attention) sweep
├── make_viz.py                # Original model figures
├── make_viz_endpoints.py      # Endpoint vs original comparison
├── make_viz_endpoints_intervals.py  # Per-interval-length viz
├── logs_endpoints_v1_itemq/   # Latest training data
├── figures/                   # Generated figures
└── old/                       # Archived experiments
```

---

## Parameters

| Parameter | CLI flag | Default | Description |
|:--|:--|:--|:--|
| ρ | `--rho` | 0.95 | Context persistence; 0 = no memory, 1 = perfect |
| σ_m | `--sigma` | 0.05 | Measurement noise SD at retrieval |
| d | `--d` | 64 | Hidden/context dimension |
| epochs | `--epochs` | 4000 | Training epochs per model |
| seed | `--seed` | 42 | Random seed |
| query-type | `--query-type` | context | `context` (c_q) or `item` (f_q) |
| variant | `--variant` | v1 | Decoder variant (`v1`, `v2`, `raw`, `learned`) |

---

## Critical Assumptions

1. **Item-level retrieval cue.** The model uses the raw item representation f_q as the query, not the accumulated context c_q. This is cognitively motivated: during retrieval, a participant sees a target frame and matches it against memory, without needing to reconstruct prior context.

2. **Endpoint-constrained attention.** Start and end reference frames are provided (as in Hu et al.), but they only constrain the attention range — not the query computation. The asymmetry arises from the item-query structure, not the endpoints.

3. **Random item vectors.** Items are isotropic Gaussian vectors (proxy for unstructured perceptual frames). The model abstracts away semantic, shot-boundary, and visual-similarity information present in the human experiment.
