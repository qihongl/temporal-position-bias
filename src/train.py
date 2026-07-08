"""
Training and Evaluation Utilities
==================================
"""

import torch
import numpy as np
from .model import TCMEncoder, AttentionDecoder
from .utils import apply_noise
from .config import (
    D_ITEM, D_CONTEXT, EPOCHS, BATCH_SIZE, LR,
    SEQ_MIN, SEQ_MAX, SEQ_TEST, TEST_POSITIONS, N_TEST_PER_POS, EVAL_BATCH,
)


def make_items(B, L, d=D_ITEM, device='cpu'):
    """Generate random item vectors."""
    return torch.randn(B, L, d, device=device)


def train_model(rho, sigma_m, epochs=EPOCHS, batch_size=BATCH_SIZE,
                d_item=D_ITEM, d_context=D_CONTEXT, seed=42,
                use_position_template=True, use_proj=True, lr=LR):
    """Train a single TCM model with attention decoder.

    Returns:
        (encoder, decoder, train_loss): final epoch MSE (scalar).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = TCMEncoder(rho, d_item, d_context, use_proj=use_proj)
    decoder = AttentionDecoder(d_context, use_position_template=use_position_template)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()

    final_loss = None
    for epoch in range(epochs):
        L = np.random.randint(SEQ_MIN, SEQ_MAX + 1)
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


def evaluate_model(encoder, decoder, sigma_m,
                   seq_test=SEQ_TEST, test_positions=TEST_POSITIONS,
                   n_test=N_TEST_PER_POS, eval_batch=EVAL_BATCH,
                   d_item=D_ITEM):
    """Evaluate trained model.

    Returns:
        dict: position_fraction -> np.array of signed errors (%)
    """
    L = seq_test
    results = {}
    for pf in test_positions:
        ti = int(pf * L)
        errors = []
        for _ in range(n_test // eval_batch):
            with torch.no_grad():
                items = make_items(eval_batch, L, d_item)
                f, c = encoder(items)
                c_q = apply_noise(c[:, ti, :], sigma_m)
                pred = decoder(c_q, c).cpu().numpy()
                errors.extend(((pred - pf) * 100).tolist())
        results[pf] = np.array(errors)
    return results


def get_attention_weights(encoder, decoder, sigma_m,
                          seq_test=SEQ_TEST, test_positions=TEST_POSITIONS,
                          d_item=D_ITEM):
    """Get attention weights at each test position."""
    L = seq_test
    items = make_items(1, L, d_item)
    attn_data = {}
    with torch.no_grad():
        f, c = encoder(items)
        for pf in test_positions:
            ti = int(pf * L)
            c_q = apply_noise(c[0, ti:ti+1, :], sigma_m)
            pos, weights = decoder(c_q, c[0:1, :, :], return_weights=True)
            attn_data[pf] = {
                'weights': weights[0].cpu().numpy(),
                'pred': pos.item() * 100,
                'true_idx': ti,
            }
    return attn_data


# Backward-compatible re-export
from .analysis import compute_summary_stats  # noqa: E402, F401
