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

    Compares query context c_q to all stored contexts c_1..c_L,
    outputs softmax-weighted average of position templates.

    Forward asymmetry emerges because rho < 1 makes c_{q+k} more
    similar to c_q than c_{q-k} for equal |k| > 0.
    """

    def __init__(self, d_context=64):
        super().__init__()
        self.query = nn.Linear(d_context, d_context, bias=False)
        self.key = nn.Linear(d_context, d_context, bias=False)
        self.raw_temp = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413) ≈ 1.0

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
        pos_template = torch.linspace(0, 1, L, device=c_all.device)
        pos = (weights * pos_template.unsqueeze(0)).sum(dim=-1)
        if return_weights:
            return pos, weights
        return pos
