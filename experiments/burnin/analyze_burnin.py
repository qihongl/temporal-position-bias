"""Analyze paired runs with the existing src.analysis mixture estimator.

The original optimizer is called unchanged. Its convergence is recorded by a
temporary wrapper; unsuccessful fits are retried from multiple starts and all
such deviations are explicitly recorded. Raw signed error is also analyzed.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from _paths import REPO_ROOT
import src.analysis as original

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
KEYS=['p20','p40','p60','p80']
SEEDS=list(range(42,62))
BOOTSTRAP_SEED=20260923


def fit_checked(errors):
    calls=[]
    def checked(fun,x0,**kwargs):
        res=scipy_minimize(fun,x0,**kwargs)
        calls.append(res)
        if not res.success or not np.isfinite(res.fun):
            starts=[[float(np.arcsin(np.clip(np.mean(errors)/80,-1,1))),np.log(k),g]
                    for k in (5,20,100) for g in (-3,0)]
            candidates=[scipy_minimize(fun,x,**kwargs) for x in starts]
            calls.extend(candidates)
            valid=[r for r in candidates if r.success and np.isfinite(r.fun)]
            if not valid:
                raise RuntimeError('Mixture MLE failed after retries')
            res=min(valid,key=lambda r:r.fun)
        return res
    previous=original.minimize
    original.minimize=checked
    try:
        mu=original.fit_memtoolbox_mu(errors)
    finally:
        original.minimize=previous
    return float(mu),{'initial_success':bool(calls[0].success),
                     'initial_message':str(calls[0].message),'attempts':len(calls),
                     'initial_objective':float(calls[0].fun)}


def asymmetry(values):
    return np.abs(values[:,:2]).mean(1)-np.abs(values[:,2:]).mean(1)


def interval(values,boot):
    v=np.asarray(values)
    samples=v[boot].mean(1)
    return {'mean':np.mean(v,axis=0).tolist(),
            'low':np.quantile(samples,.025,axis=0).tolist(),
            'high':np.quantile(samples,.975,axis=0).tolist()}


# Run analyze_burnin100.py for the active zero/100 comparison.
