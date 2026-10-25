#!/usr/bin/env python3
"""Exact full-row dominant-allocation analysis; second-best approximation diagnostic.
Exploratory fresh Gaussian scores; dominance threshold and grid fixed in script.
"""
import os; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, pandas as pd, json
from pathlib import Path
from scipy.special import softmax
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
N_LIST=[60,150,500]
TEMPS=[.02,.04,.06,.08,.10,.15,.25,.53,1.0,3.5]
NSEEDS=100; THRESHOLD=.90
r=[]
for n in N_LIST:
 for seed in range(NSEEDS):
  rng=np.random.default_rng(3320923+n*100003+seed)
  a=rng.standard_normal((n,n));np.fill_diagonal(a,-np.inf)
  vals=np.partition(a,(-2,-1),axis=1)
  gap=vals[:,-1]-vals[:,-2]
  for tau in TEMPS:
   p=softmax(a/tau,axis=1).max(axis=1)
   ap=1/(1+np.exp(-gap/tau))
   r.append(dict(N=n,seed=seed,tau=tau,fraction_single_exact=float(np.mean(p>=THRESHOLD)),fraction_single_top_two=float(np.mean(ap>=THRESHOLD)),mean_max_share=float(p.mean()),median_max_share=float(np.median(p))))
 print('dominance completed',n,flush=True)
d=pd.DataFrame(r);d.to_csv(OUT/'dominance_per_seed.csv',index=False)
s=d.groupby(['N','tau'],as_index=False)[['fraction_single_exact','fraction_single_top_two','mean_max_share','median_max_share']].mean()
s.to_csv(OUT/'dominance_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(7.3,4.35))
markers=['o','s','^']
for k,n in enumerate(N_LIST):
 x=s[s.N==n];ax.plot(x.tau,x.fraction_single_exact,marker=markers[k],lw=1.65,label=f'$N={n}$, exact');ax.plot(x.tau,x.fraction_single_top_two,ls=':',lw=1.0,alpha=.7)
ax.axhline(.5,color='gray',ls='--',lw=1.0);ax.set(xscale='log',xlabel=r'Allocation temperature $\tau$',ylabel=r'Fraction of rows with $p_{\max}\geq0.90$',ylim=(-.02,1.02));ax.legend(fontsize=9);fig.tight_layout();fig.savefig(OUT/'dominance.pdf');fig.savefig(OUT/'dominance.png',dpi=190)
(OUT/'protocol.json').write_text(json.dumps(dict(n_list=N_LIST,temperatures=TEMPS,seeds=NSEEDS,seed_rule='3320923 + N*100003 + seed',input_scores='i.i.d. standard Gaussian, no self-loops',dominance_threshold=THRESHOLD,full_row_calculation='scipy.special.softmax',top_two_formula='1/(1+exp(-gap/tau))',status='diagnostic post-hoc study, not confirmatory prediction experiment'),indent=2))
print(s[s.tau.isin([.04,.08,.15])].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
