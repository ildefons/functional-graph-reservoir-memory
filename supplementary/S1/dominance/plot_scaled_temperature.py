#!/usr/bin/env python3
"""Reproduce Figure 2 from v3.1 recorded dominance results, without new simulations."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
D=pd.read_csv(p/'dominance_summary.csv')
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(10.25,3.7),sharey=True)
styles={60:('-', 'black','o'),150:('--','#555555','s'),500:(':','#929292','^')}
for n,g in D.groupby('N'):
    line,color,mark=styles[int(n)]
    g=g.sort_values('tau');theta=g.tau*np.sqrt(2*np.log(n-1))
    axs[0].plot(g.tau,g.fraction_single_exact,line,color=color,lw=2,marker=mark,ms=3.2,label=f'N = {n}')
    axs[0].plot(g.tau,g.fraction_single_top_two,color=color,linestyle=(0,(2,3)),lw=1,alpha=.72)
    axs[1].plot(theta,g.fraction_single_exact,line,color=color,lw=2,marker=mark,ms=3.2,label=f'N = {n}')
for ax in axs:
    ax.axhline(.5,color='#888888',linestyle='-.',lw=.9)
    ax.set_ylim(0,1.01);ax.grid(alpha=.17)
    ax.set_ylabel('Fraction with maximum allocation ≥ 0.90')
axs[0].set_xlabel('Temperature τ (unit-variance scores)')
axs[1].set_xlabel('Scaled temperature θ = τ √(2 ln(N − 1))')
axs[0].set_xlim(0,.35);axs[1].set_xlim(0,1.2)
axs[0].legend(fontsize=8,loc='upper right',frameon=False)
axs[0].text(.02,.07,'Dotted: top-two approximation',transform=axs[0].transAxes,fontsize=7.5)
axs[0].set_title('(a) Raw temperature')
axs[1].set_title('(b) Extreme-value temperature scale')
fig.tight_layout(pad=1.4)
fig.savefig(p/'dominance_scaled.pdf',bbox_inches='tight')
fig.savefig(p/'dominance_scaled.png',dpi=150,bbox_inches='tight')
