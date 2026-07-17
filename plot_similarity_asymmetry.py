"""
Visualize the query-context similarity matrix and its asymmetry.
Averages across 20 seeds for smooth heatmaps.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from src.model import TCMEncoder, ConstrainedAttentionDecoder
from src.train import make_items
from src.utils import apply_noise

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
})

RHO = 0.95
SIGMA = 0.05
L = 60
N_SEEDS = 20
max_lag = 15
ks = np.arange(1, max_lag + 1)

# ---- Average raw similarity across seeds ----
raw_sim_sum = np.zeros((L, L))
fw_raw_all = np.zeros((N_SEEDS, L, max_lag))
bw_raw_all = np.zeros((N_SEEDS, L, max_lag))

for si, sd in enumerate(range(42, 42 + N_SEEDS)):
    torch.manual_seed(sd)
    items = make_items(1, L, 64)
    encoder = TCMEncoder(RHO, 64, 64, use_proj=True)
    f, c = encoder(items)
    c_norm = F.normalize(c[0], dim=-1)
    f_norm = F.normalize(f[0], dim=-1)
    sim = torch.mm(c_norm, f_norm.T).detach().numpy()
    raw_sim_sum += sim
    for q in range(L):
        for k in range(1, max_lag + 1):
            if q + k < L:
                fw_raw_all[si, q, k-1] = sim[q + k, q]
            else:
                fw_raw_all[si, q, k-1] = np.nan
            if q - k >= 0:
                bw_raw_all[si, q, k-1] = sim[q - k, q]
            else:
                bw_raw_all[si, q, k-1] = np.nan

raw_sim_avg = raw_sim_sum / N_SEEDS
fw_raw = np.nanmean(fw_raw_all, axis=0)
bw_raw = np.nanmean(bw_raw_all, axis=0)

# ---- Average learned similarity (train 10 decoders quickly) ----
learned_sim_sum = np.zeros((L, L))
fw_learned_all = np.zeros((10, L, max_lag))
bw_learned_all = np.zeros((10, L, max_lag))
N_LEARNED = 10

for si, sd in enumerate(range(42, 42 + N_LEARNED)):
    torch.manual_seed(sd)
    np.random.seed(sd)
    enc = TCMEncoder(RHO, 64, 64, use_proj=True)
    dec = ConstrainedAttentionDecoder(64)
    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=5e-4)

    for _ in range(2000):
        L2 = np.random.randint(60, 181)
        items2 = make_items(4, L2, 64)
        f2, c2 = enc(items2)
        idx_A = torch.zeros(4, dtype=torch.long)
        idx_B = torch.zeros(4, dtype=torch.long)
        idx_q = torch.zeros(4, dtype=torch.long)
        for b in range(4):
            A = np.random.randint(0, L2 - 10 - 1)
            B = np.random.randint(A + 10, L2)
            q = np.random.randint(A + 1, B)
            idx_A[b] = A; idx_B[b] = B; idx_q[b] = q
        c_q = apply_noise(f2[torch.arange(4), idx_q, :], SIGMA)
        pred = dec(c_q, c2, idx_A, idx_B)
        target = (idx_q.float() - idx_A.float()) / (idx_B.float() - idx_A.float()).clamp(min=1)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
        optimizer.step()

    with torch.no_grad():
        items3 = make_items(1, L, 64)
        f3, c3 = enc(items3)
        f3_norm = F.normalize(f3[0], dim=-1)
        q_vecs = dec.query(f3_norm)
        k_vecs = dec.key(c3[0])
        sim = torch.mm(k_vecs, q_vecs.T).detach().numpy() / dec.temperature.item()
        learned_sim_sum += sim

        for q in range(L):
            for k in range(1, max_lag + 1):
                if q + k < L:
                    fw_learned_all[si, q, k-1] = sim[q + k, q]
                else:
                    fw_learned_all[si, q, k-1] = np.nan
                if q - k >= 0:
                    bw_learned_all[si, q, k-1] = sim[q - k, q]
                else:
                    bw_learned_all[si, q, k-1] = np.nan

learned_sim_avg = learned_sim_sum / N_LEARNED
fw_learned = np.nanmean(fw_learned_all, axis=0)
bw_learned = np.nanmean(bw_learned_all, axis=0)

# ---- Figure ----
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# A: Raw similarity matrix (averaged)
ax = axes[0, 0]
im = ax.imshow(raw_sim_avg, aspect='equal', cmap='YlGnBu', origin='lower')
ax.set_xlabel('Query position q')
ax.set_ylabel('Context position i')
ax.set_title(f'A  Raw similarity: c_i · f_q  (avg over {N_SEEDS} seeds)', fontsize=11, fontweight='bold', loc='left')
plt.colorbar(im, ax=ax, shrink=0.85)

# B: Mean forward vs backward raw similarity
ax = axes[0, 1]
fw_mean_raw = np.nanmean(fw_raw, axis=0)
bw_mean_raw = np.nanmean(bw_raw, axis=0)
ax.plot(ks, fw_mean_raw, 'o-', color='#E6550D', lw=2, ms=5, label='Forward: c_{q+k} · f_q')
ax.plot(ks, bw_mean_raw, 's-', color='#3182BD', lw=2, ms=5, label='Backward: c_{q-k} · f_q')
ax.set_xlabel('|k| (lag)')
ax.set_ylabel('Mean dot product')
ax.set_title('B  Raw fw vs bw similarity (avg over q)', fontsize=11, fontweight='bold', loc='left')
ax.legend()

# C: Raw forward-backward difference as function of q
ax = axes[0, 2]
for k in [1, 2, 3, 5, 10]:
    diff_raw = fw_raw[:, k-1] - bw_raw[:, k-1]
    qs = np.arange(L)
    valid = ~(np.isnan(diff_raw))
    ax.plot(qs[valid], diff_raw[valid], 'o-', lw=1, ms=3, label=f'k={k}')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xlabel('Query position q')
ax.set_ylabel('fw − bw raw similarity')
ax.set_title('C  Raw asymmetry by query position', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)

# D: Learned similarity matrix (averaged)
ax = axes[1, 0]
im2 = ax.imshow(learned_sim_avg, aspect='equal', cmap='YlGnBu', origin='lower')
ax.set_xlabel('Query position q')
ax.set_ylabel('Context position i')
ax.set_title(f'D  Learned similarity: K(c_i)^T · Q(f_q) / T  (avg over {N_LEARNED} seeds)', fontsize=11, fontweight='bold', loc='left')
plt.colorbar(im2, ax=ax, shrink=0.85)

# E: Learned forward vs backward
ax = axes[1, 1]
fw_mean_learned = np.nanmean(fw_learned, axis=0)
bw_mean_learned = np.nanmean(bw_learned, axis=0)
ax.plot(ks, fw_mean_learned, 'o-', color='#E6550D', lw=2, ms=5, label='Forward: K(c_{q+k}) · Q(f_q)')
ax.plot(ks, bw_mean_learned, 's-', color='#3182BD', lw=2, ms=5, label='Backward: K(c_{q-k}) · Q(f_q)')
ax.set_xlabel('|k| (lag)')
ax.set_ylabel('Mean similarity score')
ax.set_title('E  Learned fw vs bw similarity (avg over q)', fontsize=11, fontweight='bold', loc='left')
ax.legend()

# F: Learned asymmetry by query position
ax = axes[1, 2]
for k in [1, 2, 3, 5, 10]:
    diff_learned = fw_learned[:, k-1] - bw_learned[:, k-1]
    qs = np.arange(L)
    valid = ~(np.isnan(diff_learned))
    ax.plot(qs[valid], diff_learned[valid], 'o-', lw=1, ms=3, label=f'k={k}')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xlabel('Query position q')
ax.set_ylabel('fw − bw learned similarity')
ax.set_title('F  Learned asymmetry by query position', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)

plt.suptitle(f'Query-Context Similarity Asymmetry  |  ρ={RHO}, σ_m={SIGMA}  |  Averaged across seeds',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
out_path = 'figures/query_context_asymmetry.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')

# Print diagnostic
print(f'\nMean forward-backward gap at k=1..5 (raw, avg over {N_SEEDS} seeds):')
for k in range(1, 6):
    print(f'  k={k}: fw={fw_mean_raw[k-1]:.4f}, bw={bw_mean_raw[k-1]:.4f}, gap={fw_mean_raw[k-1]-bw_mean_raw[k-1]:+.4f}')

print(f'\nMean forward-backward gap at k=1..5 (learned, avg over {N_LEARNED} seeds):')
for k in range(1, 6):
    print(f'  k={k}: fw={fw_mean_learned[k-1]:.4f}, bw={bw_mean_learned[k-1]:.4f}, gap={fw_mean_learned[k-1]-bw_mean_learned[k-1]:+.4f}')
