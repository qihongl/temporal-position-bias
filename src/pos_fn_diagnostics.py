"""
Diagnostic: extract learned position functions f(t) for all ρ × σ.
Saves results to disk incrementally, then plots.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from src.model import TCMEncoder, AttentionDecoder
from src.train import make_items
from src.utils import apply_noise
from src.config import TEST_POSITIONS, HUMAN_ERRORS
from src.analysis import compute_human_loss

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'legend.fontsize': 6.5, 'legend.frameon': False,
})

D = 64; EPOCHS = 4000; SEED = 42; N_T = 201
RHOS = [0.70, 0.80, 0.90, 0.95]
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25]

_rho_norm = matplotlib.colors.Normalize(vmin=0.65, vmax=1.0)
RHO_COLORS = [plt.cm.Spectral_r(_rho_norm(r)) for r in RHOS]

CACHE_DIR = os.path.join(base_dir, '.workbuddy', 'posfn_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def train_one(rho, sigma_m):
    cache_file = os.path.join(CACHE_DIR, f'fvals_r{rho:.2f}_s{sigma_m:.2f}_seed{SEED}.npz')
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        return (data['t'], data['f_vals'], data['means'].tolist(),
                float(data['asm']), float(data['hl']), float(data['mse']),
                float(data['r2']))

    torch.manual_seed(SEED); np.random.seed(SEED)
    encoder = TCMEncoder(rho, D, D)
    decoder = AttentionDecoder(D, use_position_template=False)
    opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=5e-4)
    crit = torch.nn.MSELoss()

    for _ in range(EPOCHS):
        L = np.random.randint(60, 181)
        items = make_items(32, L, D)
        fv, c = encoder(items)
        q = torch.randint(0, L, (32,))
        c_q = apply_noise(c[torch.arange(32), q, :], sigma_m)
        pred = decoder(c_q, c)
        loss = crit(pred, q.float()/L)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters())+list(decoder.parameters()), 1.0)
        opt.step()

    with torch.no_grad():
        t = torch.linspace(0, 1, N_T).unsqueeze(-1)
        f_vals = decoder.pos_fn(t).squeeze(-1).cpu().numpy()
        t_np = t.squeeze().cpu().numpy()

        eval_items = make_items(500, 100, D)
        _, eval_c = encoder(eval_items)
        errors = {}
        for pf in TEST_POSITIONS:
            ti = int(pf*100)
            cq_n = apply_noise(eval_c[:50, ti, :], sigma_m)
            p = decoder(cq_n, eval_c[:50]).cpu().numpy()
            errors[pf] = (p-pf)*100
        means = [np.mean(errors[pf]) for pf in TEST_POSITIONS]
        asm = (abs(means[0])+abs(means[1])-abs(means[2])-abs(means[3]))/2
        hl = compute_human_loss(means)
        mse = loss.item()

        lin_fit = np.polyfit(t_np, f_vals, 1)
        lin_pred = np.polyval(lin_fit, t_np)
        r2 = 1 - np.sum((f_vals-lin_pred)**2) / np.sum((f_vals-np.mean(f_vals))**2) if np.std(f_vals)>1e-8 else 0

    del encoder, decoder, opt, items, fv, c, pred, loss, eval_items, eval_c
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    np.savez(cache_file, t=t_np, f_vals=f_vals, means=np.array(means),
             asm=asm, hl=hl, mse=mse, r2=r2)
    return t_np, f_vals, means, asm, hl, mse, r2


# ---- Train all ----
print("Training models (cached to disk)...")
for i, rho in enumerate(RHOS):
    for j, sm in enumerate(SIGMAS):
        n = i*len(SIGMAS)+j+1
        print(f"  [{n}/20] ρ={rho:.2f} σ={sm:.2f}...", end=" ", flush=True)
        t, fv, means, asm, hl, mse, r2 = train_one(rho, sm)
        print(f"asm={asm:+.1f} R²={r2:.3f}")

# ---- Load cached data for plotting ----
print("\nLoading cached data for figure...")
data = {}
for rho in RHOS:
    data[rho] = {}
    for sm in SIGMAS:
        cf = os.path.join(CACHE_DIR, f'fvals_r{rho:.2f}_s{sm:.2f}_seed{SEED}.npz')
        d = np.load(cf, allow_pickle=True)
        data[rho][sm] = {'t': d['t'], 'f_vals': d['f_vals'],
                         'means': d['means'], 'asm': float(d['asm']),
                         'hl': float(d['hl']), 'mse': float(d['mse']),
                         'r2': float(d['r2'])}

# ---- FIGURE: 4×5 grid ----
fig, axes = plt.subplots(len(RHOS), len(SIGMAS),
                         figsize=(3.3*len(SIGMAS), 2.6*len(RHOS)))
fig.suptitle('Learned Position Functions f(t) — all ρ × σ (seed=42)',
             fontsize=12, fontweight='bold', y=1.005)

for i, rho in enumerate(RHOS):
    for j, sm in enumerate(SIGMAS):
        ax = axes[i, j]
        d = data[rho][sm]

        ax.plot([0,1],[0,1], '--', color='gray', lw=0.8, alpha=0.4)
        c = RHO_COLORS[i]
        ax.plot(d['t'], d['f_vals'], '-', color=c, lw=2.2, zorder=3)
        ax.fill_between(d['t'], d['t'], d['f_vals'], color=c, alpha=0.12, lw=0)

        # Annotate
        ax.text(0.52, 0.10,
                f'R²={d["r2"]:.3f}  asm={d["asm"]:+.1f}\nMSE={d["mse"]:.4f}  HL={d["hl"]:.1f}',
                transform=ax.transAxes, fontsize=5.8, va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          alpha=0.85, edgecolor='#ddd', lw=0.4))

        ax.set_xlim(0,1); ax.set_ylim(-0.18, 1.18)
        ax.set_xticks([0,0.5,1]); ax.set_yticks([0,0.5,1])

        if j == 0:
            ax.set_ylabel(f'ρ={rho}\nf(t)', fontsize=9, fontweight='bold')
        if i == 0:
            ax.set_title(f'σ={sm}', fontsize=10, fontweight='bold')
        if i == len(RHOS)-1:
            ax.set_xlabel('t', fontsize=9)

plt.tight_layout()
out = os.path.join(base_dir, 'figures_notmpl', 'pos_fn_all_params.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")

# ---- Summary grids ----
dashes = "-" * 60
print(f"\n{dashes}\nAsymmetry grid (seed={SEED}):")
print("ρ\\σ    " + "  ".join(f"{s:6.2f}" for s in SIGMAS))
for rho in RHOS:
    vals = [f"{data[rho][s]['asm']:+6.1f}" for s in SIGMAS]
    print(f"  {rho:.2f}  " + "  ".join(vals))

print(f"\nR² to linear (seed={SEED}):")
print("ρ\\σ    " + "  ".join(f"{s:6.2f}" for s in SIGMAS))
for rho in RHOS:
    vals = [f"{data[rho][s]['r2']:6.3f}" for s in SIGMAS]
    print(f"  {rho:.2f}  " + "  ".join(vals))

print(f"\nHuman loss (seed={SEED}):")
print("ρ\\σ    " + "  ".join(f"{s:6.2f}" for s in SIGMAS))
for rho in RHOS:
    vals = [f"{data[rho][s]['hl']:6.1f}" for s in SIGMAS]
    print(f"  {rho:.2f}  " + "  ".join(vals))

ha = (abs(HUMAN_ERRORS[0])+abs(HUMAN_ERRORS[1])-abs(HUMAN_ERRORS[2])-abs(HUMAN_ERRORS[3]))/2
print(f"\nHuman: asymmetry=+{ha:.1f}")
print(f"Linear template would give: f(t)=t, slope=1.0, asm ≈ +3 to +5 (depending on ρ/σ)")
