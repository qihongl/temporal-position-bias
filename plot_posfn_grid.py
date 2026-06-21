"""Plot learned position functions f(t) — all 20 (rho, sigma) combos."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'legend.fontsize': 6.5, 'legend.frameon': False,
})

from matplotlib.colors import Normalize
RHOS = [0.70, 0.80, 0.90, 0.95]
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25]
_rho_norm = Normalize(vmin=0.65, vmax=1.0)
RHO_COLORS = [plt.cm.Spectral_r(_rho_norm(r)) for r in RHOS]

CACHE = '.workbuddy/posfn_cache'

# Load all data
data = {}
for rho in RHOS:
    data[rho] = {}
    for sm in SIGMAS:
        cf = os.path.join(CACHE, f'fvals_r{rho:.2f}_s{sm:.2f}_seed42.npz')
        d = np.load(cf, allow_pickle=True)
        data[rho][sm] = {
            't': d['t'], 'f_vals': d['f_vals'],
            'means': d['means'], 'asm': float(d['asm']),
            'hl': float(d['hl']), 'mse': float(d['mse']),
            'r2': float(d['r2']),
        }

# FIGURE: 4x5 grid
fig, axes = plt.subplots(len(RHOS), len(SIGMAS),
                         figsize=(3.3 * len(SIGMAS), 2.6 * len(RHOS)))
fig.suptitle('Learned Position Functions f(t) — all ρ × σ (seed=42)',
             fontsize=12, fontweight='bold', y=1.005)

for i, rho in enumerate(RHOS):
    for j, sm in enumerate(SIGMAS):
        ax = axes[i, j]
        d = data[rho][sm]
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=0.8, alpha=0.4)
        c = RHO_COLORS[i]
        ax.plot(d['t'], d['f_vals'], '-', color=c, lw=2.2, zorder=3)
        ax.fill_between(d['t'], d['t'], d['f_vals'], color=c, alpha=0.12, lw=0)

        ax.text(0.52, 0.10,
                'R2={:.3f}  asm={:+.1f}\nMSE={:.4f}  HL={:.1f}'.format(
                    d['r2'], d['asm'], d['mse'], d['hl']),
                transform=ax.transAxes, fontsize=5.8, va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          alpha=0.85, edgecolor='#ddd', lw=0.4))

        ax.set_xlim(0, 1); ax.set_ylim(-0.18, 1.18)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        if j == 0:
            ax.set_ylabel('rho={}\nf(t)'.format(rho), fontsize=9, fontweight='bold')
        if i == 0:
            ax.set_title('sigma={}'.format(sm), fontsize=10, fontweight='bold')
        if i == len(RHOS) - 1:
            ax.set_xlabel('t', fontsize=9)

plt.tight_layout()
out = 'figures_notmpl/pos_fn_all_params.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved:', out)

# Summary grids
print()
print('Asymmetry grid:')
header = 'rho\\sigma    ' + '  '.join('{:>6.2f}'.format(s) for s in SIGMAS)
print(header)
for rho in RHOS:
    vals = '  '.join('{:>+6.1f}'.format(data[rho][s]['asm']) for s in SIGMAS)
    print('  {:.2f}  '.format(rho) + vals)

print()
print('R^2 to linear:')
print('rho\\sigma    ' + '  '.join('{:>6.2f}'.format(s) for s in SIGMAS))
for rho in RHOS:
    vals = '  '.join('{:6.3f}'.format(data[rho][s]['r2']) for s in SIGMAS)
    print('  {:.2f}  '.format(rho) + vals)

print()
print('Human loss:')
print('rho\\sigma    ' + '  '.join('{:>6.2f}'.format(s) for s in SIGMAS))
for rho in RHOS:
    vals = '  '.join('{:6.1f}'.format(data[rho][s]['hl']) for s in SIGMAS)
    print('  {:.2f}  '.format(rho) + vals)

ha = (13.41 + 7.57 - 1.91 - 7.36) / 2
print('\nHuman asymmetry: +{:.1f}'.format(ha))
print('Template model asymmetry: +3 to +5')
