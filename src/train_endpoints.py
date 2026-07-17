"""
Endpoint-Constrained Training and Evaluation
=============================================
"""
import torch
import numpy as np
from src.utils import apply_noise
from src.train import make_items
from src.config import (
    D_ITEM, D_CONTEXT, EPOCHS, BATCH_SIZE, LR,
    SEQ_MIN_ENDPOINT, SEQ_MAX, SEQ_TEST, TEST_POSITIONS,
    N_TEST_PER_POS, EVAL_BATCH, MIN_INTERVAL, INTERVAL_LENGTHS_TEST,
)
from src.model import (
    TCMEncoder, ConstrainedAttentionDecoder, EndpointConditionedDecoder,
    RawProjectionDecoder, LearnedProjectionDecoder,
)


def train_model_endpoints(rho, sigma_m, epochs=EPOCHS, batch_size=BATCH_SIZE,
                          d_item=D_ITEM, d_context=D_CONTEXT, seed=42,
                          lr=LR, min_interval=MIN_INTERVAL,
                          seq_min=SEQ_MIN_ENDPOINT, seq_max=SEQ_MAX,
                          query_type='context'):
    """Train TCM with ConstrainedAttentionDecoder.

    Each training sample draws a random interval [A, B] and target q ∈ (A, B).
    The model learns to predict relative position (q-A)/(B-A) ∈ [0, 1].
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = TCMEncoder(rho, d_item, d_context, use_proj=True)
    decoder = ConstrainedAttentionDecoder(d_context)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()
    use_item_query = (query_type == 'item')

    final_loss = None
    for epoch in range(epochs):
        L = np.random.randint(seq_min, seq_max + 1)
        items = make_items(batch_size, L, d_item)
        f, c = encoder(items)

        idx_A = torch.zeros(batch_size, dtype=torch.long)
        idx_B = torch.zeros(batch_size, dtype=torch.long)
        idx_q = torch.zeros(batch_size, dtype=torch.long)

        for b in range(batch_size):
            A = np.random.randint(0, L - min_interval - 1)
            B = np.random.randint(A + min_interval, L)
            q = np.random.randint(A + 1, B)
            idx_A[b] = A
            idx_B[b] = B
            idx_q[b] = q

        query_source = f if use_item_query else c
        c_q = apply_noise(query_source[torch.arange(batch_size), idx_q, :], sigma_m)
        pred = decoder(c_q, c, idx_A, idx_B)
        target = (idx_q.float() - idx_A.float()) / (idx_B.float() - idx_A.float()).clamp(min=1)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optimizer.step()
        final_loss = loss.item()

    return encoder, decoder, final_loss


def evaluate_model_endpoints(encoder, decoder, sigma_m,
                             seq_test=200, test_positions=TEST_POSITIONS,
                             interval_lengths=INTERVAL_LENGTHS_TEST,
                             n_test=N_TEST_PER_POS, eval_batch=EVAL_BATCH,
                             d_item=D_ITEM, query_type='context'):
    """Evaluate at 20/40/60/80% within each discrete interval length."""
    use_item_query = (query_type == 'item')
    L = seq_test
    results = {}
    for ilen in interval_lengths:
        ilen_results = {}
        for pf in test_positions:
            errors = []
            for _ in range(n_test // eval_batch):
                with torch.no_grad():
                    items = make_items(eval_batch, L, d_item)
                    f, c = encoder(items)
                    idx_A_batch = torch.zeros(eval_batch, dtype=torch.long)
                    idx_B_batch = torch.zeros(eval_batch, dtype=torch.long)
                    idx_q_batch = torch.zeros(eval_batch, dtype=torch.long)
                    for b in range(eval_batch):
                        A = np.random.randint(0, L - ilen)
                        B = A + ilen
                        q = int(pf * ilen) + A
                        idx_A_batch[b] = A
                        idx_B_batch[b] = B
                        idx_q_batch[b] = q
                    query_source = f if use_item_query else c
                    c_q = apply_noise(query_source[torch.arange(eval_batch), idx_q_batch, :], sigma_m)
                    pred = decoder(c_q, c, idx_A_batch, idx_B_batch).cpu().numpy()
                    errors.extend(((pred - pf) * 100).tolist())
            ilen_results[pf] = np.array(errors)
        results[ilen] = ilen_results
    return results


def evaluate_interval_invariance(encoder, decoder, sigma_m,
                                 d_item=D_ITEM, L_test=200, pf=0.20,
                                 n_trials=200):
    """Test error at a given relative position across varying interval lengths."""
    interval_lengths = [8, 16, 32, 64]
    results = {}
    for ilen in interval_lengths:
        errs = []
        for _ in range(n_trials):
            with torch.no_grad():
                items = make_items(1, L_test, d_item)
                f, c = encoder(items)
                A = np.random.randint(0, L_test - ilen)
                B = A + ilen
                q = int(pf * ilen) + A
                c_q = apply_noise(c[0, q:q+1, :], sigma_m)
                pred = decoder(c_q, c,
                               torch.tensor([A]), torch.tensor([B])).item()
                errs.append(pred * 100 - pf * 100)
        results[ilen] = {'mean': float(np.mean(errs)), 'std': float(np.std(errs))}
    return results


def train_model_endpoints_v2(rho, sigma_m, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             d_item=D_ITEM, d_context=D_CONTEXT, seed=42,
                             lr=LR, min_interval=MIN_INTERVAL,
                             seq_min=SEQ_MIN_ENDPOINT, seq_max=SEQ_MAX):
    """Train TCM with EndpointConditionedDecoder.

    Query is formed from [c_q, c_A, c_B] via a learned projection.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = TCMEncoder(rho, d_item, d_context, use_proj=True)
    decoder = EndpointConditionedDecoder(d_context)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()

    final_loss = None
    for epoch in range(epochs):
        L = np.random.randint(seq_min, seq_max + 1)
        items = make_items(batch_size, L, d_item)
        f, c = encoder(items)

        idx_A = torch.zeros(batch_size, dtype=torch.long)
        idx_B = torch.zeros(batch_size, dtype=torch.long)
        idx_q = torch.zeros(batch_size, dtype=torch.long)

        for b in range(batch_size):
            A = np.random.randint(0, L - min_interval - 1)
            B = np.random.randint(A + min_interval, L)
            q = np.random.randint(A + 1, B)
            idx_A[b] = A
            idx_B[b] = B
            idx_q[b] = q

        c_q = apply_noise(c[torch.arange(batch_size), idx_q, :], sigma_m)
        c_A = apply_noise(c[torch.arange(batch_size), idx_A, :], sigma_m)
        c_B = apply_noise(c[torch.arange(batch_size), idx_B, :], sigma_m)

        pred = decoder(c_q, c_A, c_B, c, idx_A, idx_B)
        target = (idx_q.float() - idx_A.float()) / (idx_B.float() - idx_A.float()).clamp(min=1)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optimizer.step()
        final_loss = loss.item()

    return encoder, decoder, final_loss


def evaluate_model_endpoints_v2(encoder, decoder, sigma_m,
                                seq_test=200, test_positions=TEST_POSITIONS,
                                interval_lengths=INTERVAL_LENGTHS_TEST,
                                n_test=N_TEST_PER_POS, eval_batch=EVAL_BATCH,
                                d_item=D_ITEM):
    """Evaluate V2 at 20/40/60/80% within each discrete interval length."""
    L = seq_test
    results = {}
    for ilen in interval_lengths:
        ilen_results = {}
        for pf in test_positions:
            errors = []
            for _ in range(n_test // eval_batch):
                with torch.no_grad():
                    items = make_items(eval_batch, L, d_item)
                    f, c = encoder(items)
                    idx_A_batch = torch.zeros(eval_batch, dtype=torch.long)
                    idx_B_batch = torch.zeros(eval_batch, dtype=torch.long)
                    idx_q_batch = torch.zeros(eval_batch, dtype=torch.long)
                    for b in range(eval_batch):
                        A = np.random.randint(0, L - ilen)
                        B = A + ilen
                        q = int(pf * ilen) + A
                        idx_A_batch[b] = A
                        idx_B_batch[b] = B
                        idx_q_batch[b] = q
                    c_q = apply_noise(c[torch.arange(eval_batch), idx_q_batch, :], sigma_m)
                    c_A = apply_noise(c[torch.arange(eval_batch), idx_A_batch, :], sigma_m)
                    c_B = apply_noise(c[torch.arange(eval_batch), idx_B_batch, :], sigma_m)
                    pred = decoder(c_q, c_A, c_B, c, idx_A_batch, idx_B_batch).cpu().numpy()
                    errors.extend(((pred - pf) * 100).tolist())
            ilen_results[pf] = np.array(errors)
        results[ilen] = ilen_results
    return results


def train_model_projection(rho, sigma_m, epochs=EPOCHS, batch_size=BATCH_SIZE,
                           d_item=D_ITEM, d_context=D_CONTEXT, seed=42,
                           lr=LR, min_interval=MIN_INTERVAL,
                           seq_min=SEQ_MIN_ENDPOINT, seq_max=SEQ_MAX,
                           learned=False, hidden=32):
    """Train TCM with projection-based decoder (raw or learned).

    Raw:     pos = (c_q-c_A)·(c_B-c_A) / ||c_B-c_A||²  (no decoder params)
    Learned: pos = MLP([c_q-c_A, c_B-c_q])               (small MLP)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = TCMEncoder(rho, d_item, d_context, use_proj=True)
    if learned:
        decoder = LearnedProjectionDecoder(d_context, hidden=hidden)
    else:
        decoder = RawProjectionDecoder(d_context)

    params = list(encoder.parameters())
    if learned:
        params += list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = torch.nn.MSELoss()

    final_loss = None
    for epoch in range(epochs):
        L = np.random.randint(seq_min, seq_max + 1)
        items = make_items(batch_size, L, d_item)
        f, c = encoder(items)

        idx_A = torch.zeros(batch_size, dtype=torch.long)
        idx_B = torch.zeros(batch_size, dtype=torch.long)
        idx_q = torch.zeros(batch_size, dtype=torch.long)

        for b in range(batch_size):
            A = np.random.randint(0, L - min_interval - 1)
            B = np.random.randint(A + min_interval, L)
            q = np.random.randint(A + 1, B)
            idx_A[b] = A
            idx_B[b] = B
            idx_q[b] = q

        c_q = apply_noise(c[torch.arange(batch_size), idx_q, :], sigma_m)
        c_A = c[torch.arange(batch_size), idx_A, :]
        c_B = c[torch.arange(batch_size), idx_B, :]

        pred = decoder(c_q, c_A, c_B)
        target = (idx_q.float() - idx_A.float()) / (idx_B.float() - idx_A.float()).clamp(min=1)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        final_loss = loss.item()

    return encoder, decoder, final_loss


def evaluate_model_projection(encoder, decoder, sigma_m,
                              seq_test=200, test_positions=TEST_POSITIONS,
                              interval_lengths=INTERVAL_LENGTHS_TEST,
                              n_test=N_TEST_PER_POS, eval_batch=EVAL_BATCH,
                              d_item=D_ITEM):
    """Evaluate projection-based decoder at multiple interval lengths."""
    L = seq_test
    results = {}
    for ilen in interval_lengths:
        ilen_results = {}
        for pf in test_positions:
            errors = []
            for _ in range(n_test // eval_batch):
                with torch.no_grad():
                    items = make_items(eval_batch, L, d_item)
                    f, c = encoder(items)
                    idx_A_batch = torch.zeros(eval_batch, dtype=torch.long)
                    idx_B_batch = torch.zeros(eval_batch, dtype=torch.long)
                    idx_q_batch = torch.zeros(eval_batch, dtype=torch.long)
                    for b in range(eval_batch):
                        A = np.random.randint(0, L - ilen)
                        B = A + ilen
                        q = int(pf * ilen) + A
                        idx_A_batch[b] = A
                        idx_B_batch[b] = B
                        idx_q_batch[b] = q
                    c_q = apply_noise(c[torch.arange(eval_batch), idx_q_batch, :], sigma_m)
                    c_A = c[torch.arange(eval_batch), idx_A_batch, :]
                    c_B = c[torch.arange(eval_batch), idx_B_batch, :]
                    pred = decoder(c_q, c_A, c_B).cpu().numpy()
                    errors.extend(((pred - pf) * 100).tolist())
            ilen_results[pf] = np.array(errors)
        results[ilen] = ilen_results
    return results
