#!/usr/bin/env python3
"""Schematic diagram of the TCM temporal position model."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'figure.facecolor': 'white',
})

fig, (ax_enc, ax_ret) = plt.subplots(1, 2, figsize=(15, 5))


def box(ax, xy, w, h, text, color='#4472C4', text_color='white', fs=10, bold=True):
    r = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.2", facecolor=color,
                       edgecolor='none', zorder=2)
    ax.add_patch(r)
    kw = dict(ha='center', va='center', fontsize=fs, zorder=3)
    if text_color:
        kw['color'] = text_color
    if bold:
        kw['fontweight'] = 'bold'
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, **kw)


def arrow(ax, x1, y1, x2, y2, lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333333',
                                lw=lw, connectionstyle='arc3,rad=0'))


def label(ax, x, y, text, fs=9):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color='#555555',
            style='italic')


def heading(ax, x, y, text):
    ax.text(x, y, text, ha='center', va='center', fontsize=13,
            fontweight='bold', color='#111111')


# ==================== ENCODING ====================
ax_enc.set_xlim(0, 11)
ax_enc.set_ylim(0, 6.5)
ax_enc.axis('off')
heading(ax_enc, 5.5, 6.1, 'Encoding')

# Items
box(ax_enc, (0.3, 4.3), 1.8, 0.8, '$x_t \\sim \\mathcal{N}(0, I_d)$', color='#666666')
label(ax_enc, 1.2, 5.2, 'Random items')
arrow(ax_enc, 2.1, 4.7, 3.2, 4.7)

# P
box(ax_enc, (3.2, 4.1), 1.8, 1.2, '$P$\n(feedforward)', color='#ED7D31')
label(ax_enc, 4.1, 5.6, 'Projection')
arrow(ax_enc, 5.0, 4.7, 5.9, 4.7)

# Normalize
box(ax_enc, (5.9, 4.2), 1.5, 1.0, 'L2-norm\n$\\|\\cdot\\|$', color='#A5A5A5')
label(ax_enc, 6.65, 5.3, '$f_t$')

# TCM recurrence
box(ax_enc, (3.2, 1.3), 5.4, 2.4, '', color='#4472C4', fs=0)
ax_enc.text(5.9, 3.4, '$c_t = \\rho\\,c_{t-1} + (1-\\rho)\\,f_t$',
            ha='center', va='center', fontsize=12, color='white',
            fontweight='bold', zorder=3, style='italic')
ax_enc.text(5.9, 2.8, '$c_0 = \\mathbf{0}$',
            ha='center', va='center', fontsize=9, color='#CCDDEE', zorder=3,
            style='italic')
label(ax_enc, 5.9, 1.8, '$\\rho = 0.95$ fixed')
arrow(ax_enc, 6.65, 4.2, 6.65, 3.7)

# Context stack
box(ax_enc, (9.5, 2.0), 0.9, 2.0, '$c_1$\n$c_2$\n$\\vdots$\n$c_L$',
    color='#4472C4', fs=8)
arrow(ax_enc, 8.6, 2.5, 9.5, 2.5)
label(ax_enc, 9.95, 4.4, 'Context\nstack C')

# ==================== RETRIEVAL ====================
ax_ret.set_xlim(0, 13.5)
ax_ret.set_ylim(0, 6.5)
ax_ret.axis('off')
heading(ax_ret, 6.75, 6.1, 'Retrieval')

# Query
box(ax_ret, (0.3, 4.5), 1.4, 0.8, '$c_q$', color='#4472C4')
arrow(ax_ret, 1.7, 4.9, 2.5, 4.9)
label(ax_ret, 2.1, 5.4, '$+ \\; \\sigma_m \\, \\varepsilon$')

# Noisy query
box(ax_ret, (2.5, 3.6), 2.2, 2.6, '', color='#4472C4', fs=0)
ax_ret.text(3.6, 5.1, 'Noisy query', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold', zorder=3)
ax_ret.text(3.6, 4.7, '$\\tilde{c}_q$', ha='center', va='center',
            fontsize=10, color='white', zorder=3, style='italic')
label(ax_ret, 3.6, 4.1, '$\\sigma_m = 0.05$')
arrow(ax_ret, 4.7, 4.9, 5.6, 4.9)

# K, Q
box(ax_ret, (5.6, 5.15), 1.4, 1.0, '$K$', color='#ED7D31', fs=11)
box(ax_ret, (5.6, 3.65), 1.4, 1.0, '$Q$', color='#ED7D31', fs=11)

# Context inputs
arrow(ax_ret, 7.0, 5.65, 7.8, 5.65)
box(ax_ret, (7.8, 5.15), 2.2, 1.0, 'Context stack  $C$',
    color='#888888', fs=9, text_color='white')
arrow(ax_ret, 7.0, 4.15, 7.8, 4.15)
box(ax_ret, (7.8, 3.65), 2.2, 1.0, 'Noisy query  $\\tilde{c}_q$',
    color='#888888', fs=9, text_color='white')

# Attention scores
arrow(ax_ret, 10.0, 5.65, 11.0, 5.65)
arrow(ax_ret, 10.0, 4.15, 11.0, 4.15)
box(ax_ret, (11.0, 3.6), 0.65, 2.6, '', color='#2E75B6', fs=0)
ax_ret.text(11.325, 4.9, '$\\alpha_i$', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold', zorder=3,
            style='italic')

# Softmax equation
ax_ret.text(8.9, 2.8, '$\\alpha = \\mathrm{softmax}\\!\\left(\\frac{K(C)\\cdot Q(\\tilde{c}_q)}{T}\\right)$',
            ha='center', va='center', fontsize=9, color='#333333',
            style='italic')
ax_ret.text(8.9, 2.35, '$T$  learned', ha='center', va='center',
            fontsize=8, color='#777777')

# Readout
arrow(ax_ret, 11.325, 3.6, 11.325, 2.35)
box(ax_ret, (7.0, 1.2), 8.65, 1.05,
    '$\\hat{q} = \\sum_i \\alpha_i \\cdot \\frac{i}{L}$        (fixed linear template)',
    color='#4472C4', fs=9.5)
label(ax_ret, 11.325, 0.85, 'Position\nestimate')

# Legend
legend_ax = fig.add_axes([0.15, 0.01, 0.7, 0.06])
legend_ax.axis('off')
legend_ax.legend(handles=[
    mpatches.Patch(facecolor='#4472C4', label='Fixed / context'),
    mpatches.Patch(facecolor='#ED7D31', label='Learned'),
    mpatches.Patch(facecolor='#666666', label='Input'),
    mpatches.Patch(facecolor='#A5A5A5', label='Normalization'),
], loc='center', ncol=4, fontsize=9, frameon=False)

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig('figures/model_schematic.png', dpi=200, bbox_inches='tight')
fig.savefig('figures/model_schematic.pdf', bbox_inches='tight')
plt.close()
print('Saved: figures/model_schematic.png')
print('Saved: figures/model_schematic.pdf')
