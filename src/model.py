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
    """Autoregressive context encoder.

    use_proj=True (default): f_t = normalize(P · x_t)  — learned projection
    use_proj=False:         f_t = normalize(x_t)       — raw items, 0 learned params
    """

    def __init__(self, rho, d_item=64, d_context=64, use_proj=True):
        super().__init__()
        self.rho = rho
        self.d_context = d_context
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Linear(d_item, d_context, bias=False)
            nn.init.orthogonal_(self.proj.weight)

    def forward(self, items):
        B, L, _ = items.shape
        if self.use_proj:
            f = F.normalize(self.proj(items), dim=-1)
        else:
            f = F.normalize(items, dim=-1)
        c = torch.zeros(B, self.d_context, device=items.device)
        contexts = []
        for t in range(L):
            c = self.rho * c + (1 - self.rho) * f[:, t]
            contexts.append(c)
        return f, torch.stack(contexts, dim=1)


class SimpleRNNEncoder(nn.Module):
    """Learned forward recurrent encoder used as a TCM control model.

    It exposes the same ``(features, contexts)`` interface as ``TCMEncoder``
    so the attention decoder and evaluation code are held fixed.
    """

    def __init__(self, d_item=64, d_context=64):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=d_item,
            hidden_size=d_context,
            nonlinearity="tanh",
            batch_first=True,
        )

    def forward(self, items):
        contexts, _ = self.rnn(items)
        return items, contexts


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


class ConstrainedAttentionDecoder(nn.Module):
    """Attention decoder constrained to interval [A, B].

    Same K/Q/T architecture as AttentionDecoder, but attention is masked to
    positions between anchor endpoints A and B, and the position template is
    normalized to the interval: (i-A)/(B-A) ∈ [0, 1].
    """

    def __init__(self, d_context=64):
        super().__init__()
        self.query = nn.Linear(d_context, d_context, bias=False)
        self.key = nn.Linear(d_context, d_context, bias=False)
        self.raw_temp = nn.Parameter(torch.tensor(0.5413))

    @property
    def temperature(self):
        return F.softplus(self.raw_temp) + 1e-6

    def forward(self, c_query, c_all, idx_A, idx_B, return_weights=False):
        """
        Args:
            c_query: (B, d) — noisy target context
            c_all:   (B, L, d) — full context stack
            idx_A:   (B,) — start indices (long)
            idx_B:   (B,) — end indices (long)
        Returns:
            pos_hat: (B,) — predicted relative position ∈ [0, 1]
        """
        B, L, d = c_all.shape
        device = c_all.device

        q_vec = self.query(c_query)
        k = self.key(c_all)
        scores = torch.bmm(k, q_vec.unsqueeze(-1)).squeeze(-1)

        idx_range = torch.arange(L, device=device).unsqueeze(0)
        mask = (idx_range >= idx_A.unsqueeze(1)) & (idx_range <= idx_B.unsqueeze(1))
        scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores / self.temperature, dim=-1)

        positions = idx_range.float()
        A = idx_A.float().unsqueeze(1)
        B = idx_B.float().unsqueeze(1)
        interval_len = (B - A).clamp(min=1)
        pos_template = (positions - A) / interval_len

        pos = (weights * pos_template).sum(dim=-1)

        if return_weights:
            return pos, weights
        return pos


class RawProjectionDecoder(nn.Module):
    """Parameter-free decoder: scalar projection of c_q onto c_A→c_B.

    pos = (c_q - c_A) · (c_B - c_A) / ||c_B - c_A||²

    No learned parameters. The encoder (TCM proj matrix) is the only
    trainable component.
    """

    def __init__(self, d_context=64):
        super().__init__()

    def forward(self, c_query, c_A, c_B):
        v = c_B - c_A
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        proj = ((c_query - c_A) * v).sum(dim=-1, keepdim=True) / v_norm_sq
        return proj.squeeze(-1)


class LearnedProjectionDecoder(nn.Module):
    """MLP on endpoint-relative context differences → position.

    Takes [c_q - c_A, c_B - c_q] (2d) through a small MLP → scalar ∈ [0, 1].
    """

    def __init__(self, d_context=64, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_context, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, c_query, c_A, c_B):
        diff = torch.cat([c_query - c_A, c_B - c_query], dim=-1)
        return self.mlp(diff).squeeze(-1)
class EndpointConditionedDecoder(nn.Module):
    """Attention decoder where endpoint contexts (c_A, c_B) shape the query.

    Query is formed by concatenating [c_q, c_A, c_B] and projecting through
    a learned linear layer (3d → d). Attention is constrained to [A, B].

    This models the human task where participants see the start and end
    frames and can compare the target against both reference points.
    """

    def __init__(self, d_context=64):
        super().__init__()
        self.query_proj = nn.Linear(3 * d_context, d_context, bias=False)
        self.key = nn.Linear(d_context, d_context, bias=False)
        self.raw_temp = nn.Parameter(torch.tensor(0.5413))

    @property
    def temperature(self):
        return F.softplus(self.raw_temp) + 1e-6

    def forward(self, c_query, c_A, c_B, c_all, idx_A, idx_B, return_weights=False):
        B, L, d = c_all.shape
        device = c_all.device

        combined = torch.cat([c_query, c_A, c_B], dim=-1)
        q_vec = self.query_proj(combined)

        k = self.key(c_all)
        scores = torch.bmm(k, q_vec.unsqueeze(-1)).squeeze(-1)

        idx_range = torch.arange(L, device=device).unsqueeze(0)
        mask = (idx_range >= idx_A.unsqueeze(1)) & (idx_range <= idx_B.unsqueeze(1))
        scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores / self.temperature, dim=-1)

        positions = idx_range.float()
        A = idx_A.float().unsqueeze(1)
        B = idx_B.float().unsqueeze(1)
        interval_len = (B - A).clamp(min=1)
        pos_template = (positions - A) / interval_len

        pos = (weights * pos_template).sum(dim=-1)

        if return_weights:
            return pos, weights
        return pos
