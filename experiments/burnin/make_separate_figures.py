"""Render each initialization in its own paper-style figure, with shared axes."""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from make_burnin_figures import (
    ROOT, OUT, POSITIONS, PX, KEYS, POSITION_COLORS, CONDITION_COLORS,
    LABELS, paper_blocks, ci_radius, plot_context, save_panel,
)


def main(burnin=100):
    if burnin != 100:
        raise ValueError("Only the 100-item comparison is active; older outputs are archived.")
    summary=json.loads((ROOT/'results/summary_burnin100.json').read_text())
    array_name='analysis_arrays_burnin100.npz'
    arrays=np.load(ROOT/'results'/array_name)
    conditions={'zero':CONDITION_COLORS['zero'],f'burnin{burnin}':'#0072B2'}
    labels={'zero':'Zero start',f'burnin{burnin}':f'{burnin}-item burn-in'}
    seeds=summary['config']['seeds'];n=len(seeds)
    pooled={condition:{p:np.concatenate([
        np.load(ROOT/'results'/condition/f'seed{s}'/'errors.npz')[key]
        for s in seeds]) for p,key in zip(POSITIONS,KEYS)}
        for condition in conditions}
    all_values=np.concatenate([v for a in pooled.values() for v in a.values()])
    error_limits=(min(-60,float(all_values.min())-5),max(60,float(all_values.max())+5))
    grid=np.linspace(*error_limits,400)
    density_max=max(gaussian_kde(v)(grid).max() for a in pooled.values() for v in a.values())*1.08
    low=min(np.min(arrays[c+'_mu'].mean(0)-ci_radius(arrays[c+'_mu'])) for c in pooled)
    high=max(np.max(arrays[c+'_mu'].mean(0)+ci_radius(arrays[c+'_mu'])) for c in pooled)
    curve_limits=(np.floor(low/5)*5-1,np.ceil(high/5)*5+1)
    magnitudes={c:np.stack([np.abs(arrays[c+'_mu'][:,:2]).mean(1),
                           np.abs(arrays[c+'_mu'][:,2:]).mean(1)],1) for c in pooled}
    bar_max=np.ceil(max(np.max(v.mean(0)+ci_radius(v)) for v in magnitudes.values())*1.18)
    blocks=paper_blocks()
    uncertainty=(f'Pointwise two-sided 95% Student-t confidence intervals across {n} trained seeds '
                 f'({n-1} degrees of freedom) for the bias curve and early/late means. '
                 'Density and CDF panels describe pooled trial distributions; no inferential intervals apply.')
    with plot_context(rc={'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans']}):
        for condition in pooled:
            data=arrays[condition+'_mu'];mag=magnitudes[condition]
            def draw(ax,panel):
                env={'np':np,'ax':ax,'POSITIONS':POSITIONS,'PX':PX,
                     'mu_means':data.mean(0),'mu_sems':ci_radius(data),'n_seeds':n,
                     'rnn_mu_means':None,'human_mu_data':None,
                     'early_mean':mag[:,0].mean(),'late_mean':mag[:,1].mean(),
                     'early_sem':ci_radius(mag[:,0]),'late_sem':ci_radius(mag[:,1]),
                     'all_errors':pooled[condition],'rdylbu_4':POSITION_COLORS,
                     'gaussian_kde':gaussian_kde}
                exec(blocks[panel],env)
                if panel==0:
                    color=conditions[condition]
                    for line in ax.lines:line.set_color(color)
                    for collection in ax.collections:
                        collection.set_color(color)
                    for artist in ax.containers:artist.set_label(labels[condition])
                    ax.set_xlabel('Target position (%)');ax.set_ylabel('Bias μ (degrees)')
                    ax.set_ylim(*curve_limits);ax.set_xlim(17,83)
                    ax.axhline(0,color='gray',lw=.7,ls='--')
                elif panel==1:
                    ax.set_xticks([0,1],['20% + 40%\nEarly','60% + 80%\nLate'])
                    ax.set_ylabel('Mean |μ| (degrees)');ax.set_ylim(0,bar_max)
                elif panel==2:
                    ax.set_xlabel('Signed error (pp)');ax.set_xlim(*error_limits);ax.set_ylim(0,density_max)
                else:
                    ax.set_xlabel('Signed error (pp)');ax.set_xlim(*error_limits);ax.set_ylim(0,1.02)
                if ax.get_legend():ax.get_legend().remove()
            caption=(f'{labels[condition]} context-constrained recurrent model. '
                     'Left to right: fitted bias μ, mean absolute early/late bias, signed-error density, and CDF. '
                     f'{n} trained seeds; 2,000 held-out sequences per position per seed (40,000 pooled trials per curve). '
                     'rho=.95, noise=.05, dimension=64. μ is in degrees after the original angular transformation; '
                     'signed error is in percentage points. All four panels reuse src/make_figures.py drawing blocks. '
                     'Corresponding panels have identical axis limits across initialization conditions. '
                     'Position colors are red=20%, orange=40%, light blue=60%, dark blue=80%. '
                     'The early/late bars retain the original paper colors. '
                     f'Source: results/{array_name} and per-seed errors.npz. '
                     'Zero start shows a pronounced early-greater-than-late bias; burn-in substantially attenuates it.')
            name='zero_start_for_burnin100' if condition=='zero' else 'burnin100'
            fig,axes=plt.subplots(1,4,figsize=(16.8,3.5))
            for panel,ax in enumerate(axes):draw(ax,panel)
            save_panel(fig,OUT/f'{name}_panels_mu.png',caption=caption,uncertainty=uncertainty,legend_ncol=5)
            plt.close(fig)
            for panel,label in enumerate(['bias_curve','early_late_bias','error_density','error_cdf']):
                f,ax=plt.subplots(figsize=(4.2,3.5));draw(ax,panel)
                save_panel(f,OUT/f'{name}_{label}.png',caption=caption+f' Individual {label} panel.',
                           uncertainty=uncertainty,legend_ncol=4)
                plt.close(f)
    print('Saved separate zero-start and burn-in figures with matched axes.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--burnin',type=int,choices=[100],default=100)
    main(parser.parse_args().burnin)
