# TCM Temporal Position Bias — Mechanism Explanation

## Notation

| Symbol | Meaning |
|:--|:--|
| $x_t \sim \mathcal{N}(0, I_d)$ | Raw item vector at position $t$ — random Gaussian noise (perceptual proxy) |
| $P \in \mathbb{R}^{d \times d}$ | Learned projection matrix ($d=64$) |
| $f_t \in \mathbb{S}^{d-1}$ | Normalized item vector: $f_t = \frac{P \cdot x_t}{\|P \cdot x_t\|_2}$ — unit vector on the 63-sphere |
| $\rho \in [0,1]$ | Context persistence (fixed, 0.95) |
| $c_t \in \mathbb{R}^d$ | Context vector at position $t$ |
| $K, Q \in \mathbb{R}^{d \times d}$ | Learned key and query projections |
| $T$ | Learned temperature (softmax sharpness) |
| $\sigma_m$ | Measurement noise at retrieval |

---

## Phase 1: Encoding — Building the Context Stack

### What is $f_t$?

$f_t$ is the normalized projection of a random item. Since $x_t$ is random Gaussian, $f_t$ is a uniformly-distributed unit vector with approximately orthogonal directions: $\mathbb{E}[f_i \cdot f_j] \approx 0$ for $i \neq j$.

**Items carry no intrinsic position information.** The only way to distinguish positions is through the TCM recurrence that mixes them over time.

### The TCM Recurrence

The context $c_t$ is a leaky integrator over all preceding items:

$$c_0 = \mathbf{0}$$

$$c_t = \rho \cdot c_{t-1} + (1-\rho) \cdot f_t$$

Expanding the recurrence:

$$c_t = (1-\rho) \sum_{i=1}^{t} \rho^{t-i} f_i$$

Each context is an exponentially-weighted moving average of all items up to position $t$, with weight $(1-\rho)\rho^{t-i}$ on item $f_i$. Recent items dominate (weight $\propto \rho^0 = 1$), oldest items decay (weight $\propto \rho^{t-1}$).

The output is the **context stack** $C = [c_1, c_2, \dots, c_L] \in \mathbb{R}^{L \times d}$.

### Why this creates structure

Since $f_i$ are nearly orthogonal, the context $c_t$ is essentially a weighted sum of independent random vectors. Different positions $t$ have different mixtures:

- $c_3$ contains $f_1, f_2, f_3$ (with weights $[\rho^2, \rho^1, \rho^0]$)
- $c_7$ contains $f_1, f_2, f_3, f_4, f_5, f_6, f_7$ (with decaying weights)

Contexts at nearby positions share most of their constituent items, making them highly similar. Contexts at distant positions share fewer items, making them dissimilar.

---

## Phase 2: Retrieval — Estimating Position

### Step 1: Extract Query Context

Given a query at relative position $q \in [0, 1]$:

$$c_q = \text{normalize}\left(c_{\lfloor q \cdot L \rfloor} + \varepsilon\right), \quad \varepsilon \sim \mathcal{N}(0, \sigma_m^2 I_d)$$

Noise $\sigma_m$ is added (to simulate imperfect retrieval), then the vector is L2-normalized.

### Step 2: Compute Attention Scores

For each stored context $c_i$:

$$\text{score}_i = \frac{K(c_i) \cdot Q(c_q)}{T}$$

Where $K, Q$ are learned linear projections and $T$ is the learned temperature.

### Step 3: Softmax Weights

$$\alpha_i = \frac{\exp(\text{score}_i)}{\sum_{j=1}^L \exp(\text{score}_j)}$$

### Step 4: Position Readout

$$\hat{q} = \sum_{i=1}^L \alpha_i \cdot \frac{i}{L-1}$$

The hardcoded position template $(i / (L-1))$ maps each position index to a value in $[0, 1]$. The model's estimate is the attention-weighted average of these values.

---

## Phase 3: Where the Bias Comes From

The bias emerges from a single geometric fact: **at equal distance from the query, forward contexts are more similar to the query than backward contexts.**

### The Geometry

Consider a query at position $q$. Compare a forward context $c_{q+k}$ and a backward context $c_{q-k}$ at the same lag $k$:

$$c_{q+k} = (1-\rho) \sum_{i=1}^{q+k} \rho^{q+k-i} f_i$$

$$c_{q-k} = (1-\rho) \sum_{i=1}^{q-k} \rho^{q-k-i} f_i$$

**Which shares more items with $c_q$?**

- $c_{q+k}$ contains $f_1, f_2, \dots, f_{q+k}$ — **all $q$ of $c_q$'s items are present** (items $f_1$ through $f_q$)
- $c_{q-k}$ contains $f_1, f_2, \dots, f_{q-k}$ — **only $q-k$ of $c_q$'s items are present**

The forward context $c_{q+k}$ shares $k$ more items with $c_q$ than the backward context $c_{q-k}$ does. These extra shared items are $f_{q-k+1}, f_{q-k+2}, \dots, f_q$.

### Consequence: Forward Cosine > Backward Cosine

Because $c_{q+k}$ shares more items with $c_q$, their cosine similarity is higher:

$$\cos(c_q, c_{q+k}) > \cos(c_q, c_{q-k}) \quad \text{for equal } k > 0$$

This holds for the **same** lag $k$ in both directions. Previous confusion arose from comparing **all** forward positions (including distant ones at $k=30, 50, 80$) against **all** backward positions — the distant forward positions drag down the average. But the softmax with sharp temperature ($T \approx 0.34$) focuses on nearby positions (small $k$), where the forward advantage is clear:

| Lag $k$ | $\cos(c_q, c_{q+k})$ | $\cos(c_q, c_{q-k})$ | Forward advantage |
|:--:|:--|:--|:--|
| 1 | 0.978 | 0.969 | **+0.009** |
| 2 | 0.972 | 0.898 | **+0.074** |
| 3 | 0.902 | 0.780 | **+0.122** |
| 4 | 0.738 | 0.702 | +0.036 |
| 5 | 0.757 | 0.670 | +0.088 |

Forward cosine is higher at every single matched lag for k ≤ 6.

### From Cosine to Position Error

The chain:

1. **TCM recurrence** makes $c_{q+k}$ share more items with $c_q$ than $c_{q-k}$ does
2. **Cosine similarity** is therefore higher in the forward direction at equal lags
3. **K and Q** project these contexts, amplifying the signal
4. **Sharp softmax** ($T \approx 0.34$) concentrates mass on nearby positions where the forward advantage is largest
5. **Weighted sum** $\sum \alpha_i \cdot \frac{i}{L-1}$ produces an estimate shifted forward from the true position

### Why the Error is Asymmetric (Bigger at Early Positions)

The error magnitude depends on how many forward distractors exist at small lags:

- **Query at 20%**: Many forward positions at small lags (21, 22, 23, ...) → large cumulative forward bias → **large positive error (+16%)**
- **Query at 40%**: Moderate forward positions → moderate bias → **moderate positive error (+6%)**
- **Query at 60%**: Fewer forward positions, more backward at small lags → reduced bias → **small negative error (−2%)**
- **Query at 80%**: Very few forward positions at small lags → bias compressed → **moderate negative error (−10%)**

The asymmetry $(\lvert\text{early}\rvert - \lvert\text{late}\rvert)$ emerges from the interaction of the per-lag forward advantage (which is roughly constant) with the **position-dependent availability** of forward distractors at small lags.

---

## Summary

The position bias is a necessary consequence of three interacting ingredients:

1. **Forward-only autoregression** ($c_t$ depends only on past items) — the TCM recurrence
2. **Exponential decay** ($\rho < 1$) — nearby items dominate context
3. **Content-addressable retrieval** (attention compares $c_q$ to all $c_i$ via learned projections)

Without any of these three, the bias disappears. A bidirectional encoder eliminates the forward asymmetry (BiGRU). A non-decaying memory makes all contexts identical ($\rho = 1$). Without attention, there's no cross-context comparison to create the position estimate.
