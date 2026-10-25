#!/usr/bin/env python3
"""Regenerate memory figures from the published per-delay data, without rerunning the experiments."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import os
S=Path(__file__).resolve().parent.parent
B=Path(os.environ.get('TCR_FIGURE_OUTPUT_DIR', str(S/'regenerated_figures')))
(B/'figures').mkdir(parents=True, exist_ok=True)
raw=pd.read_csv(S/'memory/memory_sensitivity_summary.csv')
curves=pd.read_csv(S/'memory/memory_sensitivity_by_delay.csv')
# Journal-quality printable compact 2x2 per N: fixed scale rows, dynamics columns.
colors={'tau=zero':'#0a0a0a','tau=0.08':'#a53b22','tau=0.15':'#df8b13','tau=0.53':'#9857a8','tau=3.5':'#1f77b4','gaussian':'#2a9151','cycle':'#555555'}
styles={'tau=zero':'-','tau=0.08':':','tau=0.15':'-.','tau=0.53':(0,(3,1,1,1)),'tau=3.5':'--','gaussian':(0,(5,2)),'cycle':'-'}
labels={'tau=zero':r'$\\tau=0$','tau=0.08':r'$\\tau=0.08$','tau=0.15':r'$\\tau=0.15$','tau=0.53':r'$\\tau=0.53$','tau=3.5':r'$\\tau=3.5$','gaussian':'Gaussian','cycle':'Simple cycle'}
# matplotlib MathText wants one backslash
labels={k:v.replace('\\\\tau','\\tau') for k,v in labels.items()}
for n in [60,150]:
 fig,axs=plt.subplots(2,2,figsize=(9.1,5.65),sharey=True,sharex=True)
 for i,s0 in enumerate([.6,.8]):
  for j,dyn in enumerate(['linear','tanh']):
   ax=axs[i,j];part=curves[(curves.N==n)&(curves.s0==s0)&(curves.dynamics==dyn)]
   for arm in ['tau=zero','tau=0.08','tau=0.15','tau=0.53','tau=3.5','gaussian','cycle']:
    z=part[part.arm==arm].groupby('lag',sort=True).memory_corr2.mean()
    ax.plot(z.index,z.values,color=colors[arm],ls=styles[arm],lw=1.65 if arm in ('tau=zero','cycle','tau=3.5') else 1.15,label=labels[arm])
   ax.axvline(10,color='#999999',lw=.7,ls=':')
   ax.set_xlim(0,min(2*n,180));ax.set_ylim(-.02,1.02)
   if j==0:ax.set_ylabel(f'$s_0={s0:g}$\nMemory $m(d)$')
   if i==0:ax.set_title('Linear dynamics' if j==0 else 'Tanh dynamics')
   if i==1:ax.set_xlabel('Input delay $d$ (lag zero included)')
   ax.grid(alpha=.18)
 h,l=axs[0,0].get_legend_handles_labels()
 fig.legend(h,l,loc='lower center',ncol=7,fontsize=7.5,frameon=False,bbox_to_anchor=(.5,-.006))
 fig.subplots_adjust(left=.095,right=.985,top=.93,bottom=.13,hspace=.12,wspace=.12)
 fig.savefig(B/'figures'/f'memory_sensitivity_N{n}.pdf',bbox_inches='tight')
 fig.savefig(B/'figures'/f'memory_sensitivity_N{n}.png',dpi=160,bbox_inches='tight')
 plt.close(fig)
