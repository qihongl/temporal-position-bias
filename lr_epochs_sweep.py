"""LR × Epochs grid search for TCM model (rho=0.95, sigma=0.05)."""
import torch, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import TCMEncoder, AttentionDecoder
from src.train import make_items
from src.utils import apply_noise
from src.config import TEST_POSITIONS, HUMAN_ERRORS
from src.analysis import compute_human_loss

D = 64; RHO = 0.95; SIGMA = 0.05
LRS = [5e-5, 5e-4, 5e-3, 5e-2]
EPOCHS_LIST = [1000, 2000, 4000, 8000]
SEEDS = [42, 43, 44]

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':0.8,'figure.facecolor':'white','axes.facecolor':'white',
    'legend.fontsize':7,'legend.frameon':False,
})

def train_one(lr, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    enc = TCMEncoder(RHO, D, D, use_proj=True)
    dec = AttentionDecoder(D, use_position_template=True)
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=lr)
    crit = torch.nn.MSELoss()

    loss_curve = []
    for epoch in range(epochs):
        L = np.random.randint(60, 181)
        items = make_items(32, L, D)
        _, c = enc(items)
        q = torch.randint(0, L, (32,))
        c_q = apply_noise(c[torch.arange(32), q, :], SIGMA)
        pred = dec(c_q, c)
        loss = crit(pred, q.float()/L)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()), 1.0)
        opt.step()
        if epoch % max(1, epochs // 40) == 0:
            loss_curve.append((epoch, loss.item()))
    loss_curve.append((epochs-1, loss.item()))

    with torch.no_grad():
        eval_items = make_items(500, 100, D)
        _, eval_c = enc(eval_items)
        means = []
        for pf in TEST_POSITIONS:
            ti = int(pf*100)
            cq_n = apply_noise(eval_c[:50, ti, :], SIGMA)
            p = dec(cq_n, eval_c[:50]).cpu().numpy()
            means.append(float(np.mean((p-pf)*100)))
        asm = (abs(means[0])+abs(means[1])-abs(means[2])-abs(means[3]))/2
        hl = compute_human_loss(means)
        mae = float(np.mean([abs(np.mean((cc-pf)*100)) for cc, pf in 
                    zip([dec(apply_noise(eval_c[:50, int(pf*100), :], SIGMA),
                         eval_c[:50]).cpu().numpy() for pf in TEST_POSITIONS],
                        TEST_POSITIONS)]))
        # Recompute MAE properly
        mae2 = float(np.mean([abs(m) for m in means]))

    return loss_curve, means, asm, hl, mae2

# ---- Run grid ----
print(f"Grid search: {len(LRS)} LRs x {len(EPOCHS_LIST)} epochs x {len(SEEDS)} seeds = {len(LRS)*len(EPOCHS_LIST)*len(SEEDS)} runs")
results = {}
for lr in LRS:
    for ep in EPOCHS_LIST:
        key = (lr, ep)
        results[key] = {'curves': [], 'means': [], 'asms': [], 'hls': [], 'maes': []}
        for seed in SEEDS:
            curve, means, asm, hl, mae = train_one(lr, ep, seed)
            results[key]['curves'].append(curve)
            results[key]['means'].append(means)
            results[key]['asms'].append(asm)
            results[key]['hls'].append(hl)
            results[key]['maes'].append(mae)
        avg_hl = np.mean(results[key]['hls'])
        avg_mae = np.mean(results[key]['maes'])
        avg_asm = np.mean(results[key]['asms'])
        print(f"lr={lr:.0e} ep={ep:5d}: HL={avg_hl:5.1f} MAE={avg_mae:5.1f}% asm={avg_asm:+5.1f}")

# ---- HEATMAP FIGURE ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Build grids
hl_grid = np.zeros((len(LRS), len(EPOCHS_LIST)))
mae_grid = np.zeros((len(LRS), len(EPOCHS_LIST)))
asm_grid = np.zeros((len(LRS), len(EPOCHS_LIST)))
best_hl = float('inf')
best_key = None
for i, lr in enumerate(LRS):
    for j, ep in enumerate(EPOCHS_LIST):
        hl_grid[i, j] = np.mean(results[(lr, ep)]['hls'])
        mae_grid[i, j] = np.mean(results[(lr, ep)]['maes'])
        asm_grid[i, j] = np.mean(results[(lr, ep)]['asms'])
        if hl_grid[i, j] < best_hl:
            best_hl = hl_grid[i, j]
            best_key = (i, j)

# Panel A: Human loss heatmap
ax = axes[0]
im = ax.imshow(hl_grid, aspect='auto', cmap='RdYlGn_r',
               extent=[EPOCHS_LIST[0]-400, EPOCHS_LIST[-1]+400, LRS[-1]*1.5, LRS[0]*0.5],
               origin='upper')
ax.set_yscale('log')
ax.set_xlabel('Epochs'); ax.set_ylabel('Learning rate')
ax.set_title('Human loss (lower = better)', fontsize=11, fontweight='bold', loc='left')
plt.colorbar(im, ax=ax, shrink=0.85, label='L1 distance to human')
for i in range(len(LRS)):
    for j in range(len(EPOCHS_LIST)):
        ax.text(EPOCHS_LIST[j], LRS[i], f'{hl_grid[i,j]:.1f}', ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white' if hl_grid[i,j] > (hl_grid.max()+hl_grid.min())/2 else 'black')
ax.scatter(EPOCHS_LIST[best_key[1]], LRS[best_key[0]], marker='*', s=300,
           color='#2ca02c', edgecolors='black', linewidth=1, zorder=5)

# Panel B: Task accuracy heatmap (MAE)
ax = axes[1]
im = ax.imshow(mae_grid, aspect='auto', cmap='RdYlGn_r',
               extent=[EPOCHS_LIST[0]-400, EPOCHS_LIST[-1]+400, LRS[-1]*1.5, LRS[0]*0.5],
               origin='upper')
ax.set_yscale('log')
ax.set_xlabel('Epochs'); ax.set_ylabel('Learning rate')
ax.set_title('Task accuracy — MAE (%) (lower = better)', fontsize=11, fontweight='bold', loc='left')
plt.colorbar(im, ax=ax, shrink=0.85, label='Mean absolute error (%)')
for i in range(len(LRS)):
    for j in range(len(EPOCHS_LIST)):
        ax.text(EPOCHS_LIST[j], LRS[i], f'{mae_grid[i,j]:.1f}', ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white' if mae_grid[i,j] > (mae_grid.max()+mae_grid.min())/2 else 'black')

plt.suptitle(f'LR x Epochs Grid — TCM (rho=0.95, sigma=0.05, avg 3 seeds) — Best HL={best_hl:.1f}',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out = 'figures/lr_epochs_heatmap.png'
os.makedirs('figures', exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")
print(f"\nBest config: lr={LRS[best_key[0]]:.0e}, epochs={EPOCHS_LIST[best_key[1]]}, HL={best_hl:.1f}")
print(f"Default config: lr=5e-4, epochs=2000, HL={hl_grid[1,1]:.1f}")

# ---- Error curves at best config ----
print("\nError patterns at best config (3 seeds):")
best_lr = LRS[best_key[0]]; best_ep = EPOCHS_LIST[best_key[1]]
for seed, means in zip(SEEDS, results[(best_lr, best_ep)]['means']):
    print(f"  seed={seed}: {[f'{m:+.1f}' for m in means]}")

print(f"\nHuman: {[f'{h:+.1f}' for h in HUMAN_ERRORS]}")
