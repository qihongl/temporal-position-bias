"""Ablation study: learning curves + final error patterns for 5 parameter variants."""
import torch, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from src.train import make_items
from src.utils import apply_noise
from src.model import TCMEncoder
from src.analysis import compute_human_loss

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'legend.fontsize': 8, 'legend.frameon': False,
})

D = 64; RHO = 0.95; SIGMA = 0.05; EPOCHS = 4000; SEED = 42
LOG_EVERY = 100

COLORS = ['#1F78B4', '#33A02C', '#E31A1C', '#FF7F00', '#6A3D9A']

def train_variant(name, variant):
    torch.manual_seed(SEED); np.random.seed(SEED)
    enc = TCMEncoder(RHO, D, D)

    if variant == 'full':
        from src.model import AttentionDecoder
        dec = AttentionDecoder(D, use_position_template=True)
    elif variant == 'no_P':
        from src.model import AttentionDecoder
        dec = AttentionDecoder(D, use_position_template=True)
        enc.proj.weight.data = torch.eye(D)
        enc.proj.weight.requires_grad = False
    elif variant == 'no_KQ':
        class RawDec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.raw_temp = torch.nn.Parameter(torch.tensor(0.5413))
                self.use_position_template = True
            @property
            def temperature(self):
                return torch.nn.functional.softplus(self.raw_temp) + 1e-6
            def forward(self, c_q, c_all, return_weights=False):
                B, L, d = c_all.shape
                scores = torch.bmm(c_all, c_q.unsqueeze(-1)).squeeze(-1)
                w = torch.nn.functional.softmax(scores / self.temperature, dim=-1)
                tpl = torch.linspace(0, 1, L, device=c_all.device)
                pos = (w * tpl.unsqueeze(0)).sum(dim=-1)
                return (pos, w) if return_weights else pos
        dec = RawDec()
    elif variant == 'no_P_KQ':
        class RawDec2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.raw_temp = torch.nn.Parameter(torch.tensor(0.5413))
                self.use_position_template = True
            @property
            def temperature(self):
                return torch.nn.functional.softplus(self.raw_temp) + 1e-6
            def forward(self, c_q, c_all, return_weights=False):
                B, L, d = c_all.shape
                scores = torch.bmm(c_all, c_q.unsqueeze(-1)).squeeze(-1)
                w = torch.nn.functional.softmax(scores / self.temperature, dim=-1)
                tpl = torch.linspace(0, 1, L, device=c_all.device)
                pos = (w * tpl.unsqueeze(0)).sum(dim=-1)
                return (pos, w) if return_weights else pos
        dec = RawDec2()
        enc.proj.weight.data = torch.eye(D)
        enc.proj.weight.requires_grad = False
    elif variant == 'zero_param':
        class FixedDec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.use_position_template = True
            def forward(self, c_q, c_all, return_weights=False):
                B, L, d = c_all.shape
                scores = torch.bmm(c_all, c_q.unsqueeze(-1)).squeeze(-1)
                w = torch.nn.functional.softmax(scores, dim=-1)
                tpl = torch.linspace(0, 1, L, device=c_all.device)
                pos = (w * tpl.unsqueeze(0)).sum(dim=-1)
                return (pos, w) if return_weights else pos
        dec = FixedDec()
        enc.proj.weight.data = torch.eye(D)
        enc.proj.weight.requires_grad = False

    params_list = [p for p in list(enc.parameters()) + list(dec.parameters()) if p.requires_grad]
    n_params = sum(p.numel() for p in params_list)
    opt = torch.optim.Adam(params_list, lr=5e-4) if params_list else None
    crit = torch.nn.MSELoss()

    losses = []
    for epoch in range(EPOCHS):
        L = np.random.randint(60, 181)
        items = make_items(32, L, D)
        _, c = enc(items)
        q = torch.randint(0, L, (32,))
        c_q = apply_noise(c[torch.arange(32), q, :], SIGMA)
        pred = dec(c_q, c)
        loss = crit(pred, q.float()/L)
        if opt is not None:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params_list, 1.0)
            opt.step()
        if epoch % LOG_EVERY == 0:
            losses.append((epoch, loss.item()))

    with torch.no_grad():
        eval_items = make_items(500, 100, D)
        _, eval_c = enc(eval_items)
        means = []
        for pf in [0.20, 0.40, 0.60, 0.80]:
            ti = int(pf * 100)
            cq_n = apply_noise(eval_c[:50, ti, :], SIGMA)
            p = dec(cq_n, eval_c[:50]).cpu().numpy()
            means.append(float(np.mean((p - pf) * 100)))
        asm = (abs(means[0]) + abs(means[1]) - abs(means[2]) - abs(means[3])) / 2
        hl = compute_human_loss(means)

    return losses, means, asm, hl, n_params, dec


variants = [
    ('Full P+K+Q+T\n12,289 params', 'full'),
    ('No P\n8,193 params', 'no_P'),
    ('No K, Q\n4,097 params', 'no_KQ'),
    ('No P, K, Q\n1 param (T)', 'no_P_KQ'),
    ('Zero param\nraw cosine', 'zero_param'),
]

print("Training 5 variants...")
results = {}
for i, (vname, vcode) in enumerate(variants):
    print(f"  [{i+1}/5] {vname.split(chr(10))[0]}...", end=" ", flush=True)
    losses, means, asm, hl, nparam, dec = train_variant(vname, vcode)
    results[vname] = {'losses': losses, 'means': means, 'asm': asm,
                      'hl': hl, 'n_params': nparam}
    print(f"loss={losses[-1][1]:.4f} asm={asm:+.1f} HL={hl:.1f}")

# ---- FIGURE ----
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# A: Learning curves
ax = fig.add_subplot(gs[0, :2])
for i, (vname, _) in enumerate(variants):
    d = results[vname]
    epochs, ls = zip(*d['losses'])
    ax.plot(epochs, ls, '-', color=COLORS[i], lw=1.8,
            label=vname.replace('\n', ' '), alpha=0.9)
ax.axhline(0.0833, color='gray', ls=':', lw=1, alpha=0.5,
           label='Constant pred. MSE')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training MSE')
ax.set_title('A  Learning curves (seed=42)', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7.5, loc='upper right', ncol=2)
ax.set_yscale('log')
ax.set_xlim(0, EPOCHS)

# B: Final error patterns
ax = fig.add_subplot(gs[0, 2])
x_pos = [20, 40, 60, 80]
for i, (vname, _) in enumerate(variants):
    d = results[vname]
    ax.plot(x_pos, d['means'], 'o-', color=COLORS[i], lw=1.8, ms=6,
            label=vname.replace('\n', ' '), alpha=0.9)
ax.plot(x_pos, [13.41, 7.57, -1.91, -7.36], 'D-', color='black',
        lw=2.5, ms=9, label='Human', zorder=10)
ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.4)
ax.set_xlabel('True position (%)')
ax.set_ylabel('Signed error (%)')
ax.set_title('B  Error vs position', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=6.5, loc='lower left')
ax.set_xlim(10, 90)

# C: Asymmetry bar chart
ax = fig.add_subplot(gs[1, 0])
names_short = [v[0].split('\n')[0] for v in variants]
asm_vals = [results[v[0]]['asm'] for v in variants]
bars = ax.bar(range(len(asm_vals)), asm_vals, color=COLORS, edgecolor='white', lw=0.8)
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axhline(5.9, color='black', ls=':', lw=1.5, label='Human: +5.9')
ax.set_xticks(range(len(names_short)))
ax.set_xticklabels(names_short, fontsize=8)
ax.set_ylabel('Asymmetry (%)')
ax.set_title('C  Asymmetry', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=7)
for bar, v in zip(bars, asm_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            v + 0.2 if v >= 0 else v - 0.8,
            f'{v:+.1f}', ha='center', fontsize=9, fontweight='bold')

# D: Human loss
ax = fig.add_subplot(gs[1, 1])
hl_vals = [results[v[0]]['hl'] for v in variants]
bars = ax.bar(range(len(hl_vals)), hl_vals, color=COLORS, edgecolor='white', lw=0.8)
ax.set_xticks(range(len(names_short)))
ax.set_xticklabels(names_short, fontsize=8)
ax.set_ylabel('Human loss (lower = better)')
ax.set_title('D  Human loss', fontsize=11, fontweight='bold', loc='left')
for bar, v in zip(bars, hl_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
            f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')

# E: Summary table
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
table_data = [
    ['Variant', 'Params', 'Asym', 'HL'],
    ['Full P+K+Q+T', '12,289', '+4.3', '14.8'],
    ['No P', '8,193', '+3.4', '12.9'],
    ['No K,Q', '4,097', '+0.9', '41.1'],
    ['No P,K,Q', '1', '+1.3', '41.0'],
    ['Zero param', '0', '+0.2', '48.7'],
]
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.28, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
for key, cell in table.get_celld().items():
    cell.set_edgecolor('#ddd')
    cell.set_linewidth(0.3)
    if key[0] == 0:
        cell.set_facecolor('#F1EFE8')
        cell.set_text_props(weight='bold')
    elif key[1] == 0:
        cell.set_text_props(ha='left')
        cell._loc = 'left'
ax.set_title('E  Summary', fontsize=11, fontweight='bold', loc='left')

plt.suptitle('Parameter Ablation — TCM (rho=0.95, sigma=0.05)',
             fontsize=14, fontweight='bold', y=0.995)
out_dir = os.path.join(base_dir, 'figures')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'ablation_learned_params.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")
