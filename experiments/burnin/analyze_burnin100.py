"""Analyze paired zero-start and 100-item burn-in runs without archived inputs."""
import csv
import json
import numpy as np
from analyze_burnin import ROOT,OUT,KEYS,BOOTSTRAP_SEED,fit_checked,asymmetry,interval


def main():
    cfg=json.loads((OUT/'config_burnin100.json').read_text())
    seeds=cfg['seeds']
    arrays={}
    diagnostics=[]
    rows=[]
    for condition,burnin in [('zero',0),('burnin100',100)]:
        collected={k:[] for k in ['mu','signed','legacy_mu','legacy_signed','mae','mse','norms']}
        for seed in seeds:
            with np.load(OUT/condition/f'seed{seed}'/'errors.npz') as data:
                mu=[];legacy_mu=[]
                for prefix,container in [('',mu),('legacy_',legacy_mu)]:
                    for key in KEYS:
                        estimate,status=fit_checked(data[prefix+key])
                        container.append(estimate)
                        diagnostics.append({'condition':condition,'seed':seed,'eval_burnin':burnin,
                                            'position':prefix+key,**status})
                signed=[float(data[k].mean()) for k in KEYS]
                legacy_signed=[float(data['legacy_'+k].mean()) for k in KEYS]
                errors=np.concatenate([data[k] for k in KEYS])
                mae=float(np.abs(errors).mean());mse=float(np.square(errors).mean())
                values={'mu':mu,'signed':signed,'legacy_mu':legacy_mu,'legacy_signed':legacy_signed,
                        'mae':mae,'mse':mse,'norms':data['context_norm_squared']}
                for key in collected:collected[key].append(values[key])
                for j,key in enumerate(KEYS):
                    rows.append({'seed':seed,'training_burnin':burnin,'evaluation_burnin':burnin,
                                 'position':key,'mu_degrees':mu[j],'mean_signed_error_pp':signed[j],
                                 'mae_pp':mae,'mse_pp2':mse})
        arrays[condition]={k:np.array(v) for k,v in collected.items()}
        # Preserve the baseline's original float32 reduction arithmetic.
        if condition == 'zero':
            for metric in ['signed','legacy_signed','mae','mse']:
                arrays[condition][metric]=arrays[condition][metric].astype(np.float32)
        print(condition,'asymmetry',float(asymmetry(arrays[condition]['mu']).mean()),flush=True)
    boot=np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(seeds),(20000,len(seeds)))
    stats={c:{**{m:interval(a[m],boot) for m in ['mu','signed','mae','mse']},
              **{m+'_asymmetry':interval(asymmetry(a[m]),boot)
                 for m in ['mu','signed','legacy_mu','legacy_signed']}} for c,a in arrays.items()}
    a,b=arrays['burnin100'],arrays['zero']
    comparison={**{m+'_asymmetry':interval(asymmetry(a[m])-asymmetry(b[m]),boot)
                    for m in ['mu','signed','legacy_mu','legacy_signed']},
                **{m:interval(a[m]-b[m],boot) for m in ['mae','mse']}}
    a0=asymmetry(b['mu']);a100=asymmetry(a['mu'])
    ratios=100*(1-a100[boot].mean(1)/a0[boot].mean(1))
    result={'config':cfg,'conditions':stats,'paired_comparisons':{'burnin100_minus_zero':comparison},
            'reduction_percent_vs_zero':{'mean':float(100*(1-a100.mean()/a0.mean())),
                                        'low':float(np.quantile(ratios,.025)),
                                        'high':float(np.quantile(ratios,.975))},
            'fits':{'count':len(diagnostics),'initial_failures':sum(not d['initial_success'] for d in diagnostics)},
            'bootstrap':{'seed':BOOTSTRAP_SEED,'draws':20000,'unit':'paired trained seed',
                         'interval':'two-sided 95% percentile'}}
    (OUT/'summary_burnin100.json').write_text(json.dumps(result,indent=2))
    (OUT/'mixture_fit_diagnostics_burnin100.json').write_text(json.dumps(diagnostics,indent=2))
    np.savez_compressed(OUT/'analysis_arrays_burnin100.npz',
                        **{c+'_'+m:a for c,v in arrays.items() for m,a in v.items()})
    with (OUT/'per_seed_metrics_burnin100.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print('Saved zero/100 comparison; archived experiments were not read.',flush=True)


if __name__=='__main__':
    main()
