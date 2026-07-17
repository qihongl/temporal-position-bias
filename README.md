# TCM Temporal Position Model

A minimal neural network showing that asymmetric errors in temporal position judgment emerge from a forward-autoregressive context encoder combined with content-addressable retrieval — no hand-specified asymmetry.

## Model

**Encoding.** Items `x_t ~ N(0, I)` pass through a learned projection `P` and are normalized: `f_t = normalize(P · x_t)`. A context vector accumulates via the TCM recurrence:

```
c_0 = 0
c_t = ρ · c_{t-1} + (1−ρ) · f_t
```

`c_t` is an exponentially-weighted sum of all items up to `t` — recent items dominate, old items decay.

**Retrieval.** Given a query at position `q`, the model takes `c_q` (adds Gaussian noise `σ_m`, normalizes) and computes attention over all stored contexts:

```
scores_i = K(c_i) · Q(c_q) / T
α = softmax(scores)
pôs = Σ_i α_i · (i / L)
```

`K`, `Q` (64×64) and `T` (scalar) are learned. The position template `i/L` is fixed. Loss is MSE against the true relative position.

**Why bias emerges.** Because `c_t` depends only on past items, `c_{q+k}` shares more constituent items with `c_q` than `c_{q-k}` does at equal lag. Forward contexts are geometrically more similar to the query, so attention shifts forward. The shift is larger at early positions (more room ahead) than late ones, producing the asymmetric error pattern.

Learnable: `P`, `K`, `Q`, `T` (~12K params). Fixed: `ρ = 0.95`, `σ_m = 0.05`, TCM recurrence, softmax, position template.

## Usage

```bash
pip install -r requirements.txt

# Train model with global attention (c_q query, 5 seeds)
python sweep.py --rhos "0.95" --sigmas "0.05" \
    --seeds "42,43,44,45,46" --epochs 4000 --d 64 --batch 32

# Generate analysis panels
python make_panels.py
```
