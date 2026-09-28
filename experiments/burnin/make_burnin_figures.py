"""Shared plotting helpers for the active zero/100-item comparison."""
from pathlib import Path
import ast
import json
import os
import sys
from _paths import ROOT, REPO_ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,t

sys.path.insert(0,str(ROOT/'plot_style/scripts'))
from plot_style import plot_context,save_panel

OUT=ROOT/'figures_burnin'
KEYS=['p20','p40','p60','p80']
POSITIONS=[.2,.4,.6,.8]
PX=[20,40,60,80]
POSITION_COLORS=['#D73027','#F46D43','#74ADD1','#4575B4']
CONDITION_COLORS={'zero':'#E6550D','burnin100':'#0072B2'}
LABELS={'zero':'Zero start','burnin100':'100-item burn-in'}


def paper_blocks():
    tree=ast.parse((REPO_ROOT/'src/make_figures.py').read_text())
    loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
              and n.target.id=='rho')
    starts=[]
    for i,node in enumerate(loop.body):
        if (isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
            and node.targets[0].id=='ax' and isinstance(node.value,ast.Subscript)
            and isinstance(node.value.value,ast.Name) and node.value.value.id=='axes'):
            starts.append(i)
    blocks=[]
    for panel,start in enumerate(starts):
        stop=starts[panel+1] if panel+1<len(starts) else len(loop.body)
        nodes=[]
        for node in loop.body[start+1:stop]:
            if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Attribute)
                and isinstance(node.value.func.value,ast.Name)
                and node.value.func.value.id=='fig'):
                break
            nodes.append(node)
        blocks.append(compile(ast.Module(body=nodes,type_ignores=[]),
                              str(REPO_ROOT/'src/make_figures.py'),'exec'))
    assert len(blocks)==4
    return blocks


def ci_radius(a):
    return t.ppf(.975,len(a)-1)*np.std(a,axis=0,ddof=1)/np.sqrt(len(a))


# Run make_separate_figures.py to regenerate the active figures.
