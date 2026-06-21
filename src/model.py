"""
TCM Model Architecture
=======================
Encoder: c_t = rho * c_{t-1} + (1-rho) * f_t
Decoder: attention over all stored contexts -> softmax-weighted position
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCMEncoder(nn.Module):
    """Autoregressive context encoder."""

    def __init__(self, rho, d_item=64, d_context=64):
        super().__init__()
        self.rho = rho
        self.d_context = d_context
        self.proj = nn.Linear(d_item, d_context, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, items):
        B, L, _ = items.shape
        f = F.normalize(self.proj(items), dim=-1)
        c = torch.zeros(B, self.d_context, device=items.device)
        contexts = []
        for t in range(L):
            c = self.rho * c + (1 - self.rho) * f[:, t]
            contexts.append(c)
        return f, torch.stack(contexts, dim=1)


class AttentionDecoder(nn.Module):
    """
    Attention-based position decoder.

    Compares query context c_q to all stored contexts c_1..c_L.
    Two readout modes controlled by use_position_template:

    - True (default): pos = Σ_i α_i · (i/L)
      Hardcoded linear position template [0, 1]. The decoder only learns
      where to attend; position values are fixed.

    - False: pos = Σ_i α_i · f(i / (L−1))
      Learned position function f: [0, 1] → ℝ. A small MLP replaces the
      hardcoded (i/L) template with a learnable nonlinear mapping. The
      attention mechanism is identical — only the position VALUES change.
      No output activation; MSE loss constrains f to roughly [0, 1].

    Forward asymmetry emerges because rho < 1 makes c_{q+k} more
    similar to c_q than c_{q-k} for equal |k| > 0.
    """

    def __init__(self, d_context=64, use_position_template=True,
                 pos_fn_hidden=32):
        super().__init__()
        self.use_position_template = use_position_template
        self.query = nn.Linear(d_context, d_context, bias=False)
        self.key = nn.Linear(d_context, d_context, bias=False)
        self.raw_temp = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413) ≈ 1.0

        if not use_position_template:
            # Learned position function f: t ∈ [0, 1] → position value
            # Small MLP with sigmoid output to stay in [0, 1]
            # Initialised with small weights → roughly f(t) ≈ 0.5
            # then converges to a potentially nonlinear position scale
            self.pos_fn = nn.Sequential(
                nn.Linear(1, pos_fn_hidden),
                nn.ReLU(),
                nn.Linear(pos_fn_hidden, pos_fn_hidden),
                nn.ReLU(),
                nn.Linear(pos_fn_hidden, 1),
            )
            # No sigmoid — let f(t) learn its own range.
            # MSE loss naturally constrains output to roughly [0, 1]

    @property
    def temperature(self):
        """Strictly positive temperature via softplus (avoids division by zero)."""
        return F.softplus(self.raw_temp) + 1e-6

    def forward(self, c_query, c_all, return_weights=False):
        B, L, d = c_all.shape
        q = self.query(c_query)
        k = self.key(c_all)
        scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)
        weights = F.softmax(scores / self.temperature, dim=-1)

        if self.use_position_template:
            pos_template = torch.linspace(0, 1, L, device=c_all.device)
            pos = (weights * pos_template.unsqueeze(0)).sum(dim=-1)
        else:
            # Learned position function f(t) where t ∈ [0, 1]
            t = torch.linspace(0, 1, L, device=c_all.device).unsqueeze(-1)  # (L, 1)
            pos_vals = self.pos_fn(t).squeeze(-1)  # (L,)
            pos = (weights * pos_vals.unsqueeze(0)).sum(dim=-1)

        if return_weights:
            return pos, weights
        return pos
