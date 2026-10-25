#!/usr/bin/env python3
"""Exact random signed functional graph polynomial and scaling checks.
An explicitly non-leaky LINEAR scalar-input zero-temperature TCR limit only.
No numerical Krylov-rank inference is used for N>=60.
"""
from collections import deque
from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
from sympy import divisors, totient

def graph_stats(successor, sign):
    n=len(successor); indeg=np.bincount(successor,minlength=n).astype(int)
    q=deque(np.flatnonzero(indeg==0).tolist()); order=[]
    while q:
        i=q.popleft();order.append(i);j=int(successor[i]);indeg[j]-=1
        if indeg[j]==0:q.append(j)
    cyc=np.flatnonzero(indeg>0); seen=np.zeros(n,bool); specs=[]
    for i in cyc:
        if seen[i]:continue
        j=int(i);prod=1;L=0
        while not seen[j]:
            seen[j]=True;prod*=int(sign[j]);L+=1;j=int(successor[j])
        specs.append((L,prod))
    height=np.zeros(n,int)
    for i in reversed(order):height[i]=height[successor[i]]+1
    h=int(height.max());c=int(len(cyc))
    # cyclotomic polynomial factorization over Q:
    # x^L - 1: Phi_d, d|L; x^L + 1: Phi_d, d|2L with d not dividing L
    included=set()
    for L,prod in specs:
        if prod==1: included.update(int(d) for d in divisors(L))
        else: included.update(int(d) for d in divisors(2*L) if L%d!=0)
    cyclo_degree=sum(int(totient(d)) for d in included)
    rank_exact=h+cyclo_degree
    assert h<=rank_exact<=h+c
    return dict(N=n,C=c,H=h,degree_cycle_part=cyclo_degree,generic_krylov_rank_exact=rank_exact,bound_C_plus_H=h+c,cycles=len(specs),cycle_specs=';'.join(f'{L}:{prod}' for L,prod in specs))

def one_graph(n,rng):
    idx=np.arange(n); src=rng.integers(0,n-1,size=n); src+=(src>=idx)
    signs=rng.choice(np.array([-1,1]),size=n)
    return src,signs

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=100);p.add_argument('--out',default='.')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ns=[60,100,150,250,500,1000,2000];rows=[]
    for n in ns:
        for seed in range(args.seeds):
            rng=np.random.default_rng(9715511+100003*n+seed)
            suc,sgn=one_graph(n,rng)
            r=graph_stats(suc,sgn);r['seed']=seed;rows.append(r)
        print('graph',n,'completed',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'graph_exact_per_seed.csv',index=False)
    group=df.groupby('N',as_index=False).agg(seeds=('seed','size'),mean_H=('H','mean'),mean_C=('C','mean'),mean_rank=('generic_krylov_rank_exact','mean'),mean_upper=('bound_C_plus_H','mean'),sd_rank=('generic_krylov_rank_exact','std'),mean_cycle_count=('cycles','mean'))
    for c in ['mean_H','mean_C','mean_rank','mean_upper']:
        group[c+'_over_sqrtN']=group[c]/np.sqrt(group.N)
    group.to_csv(out/'graph_scaling_summary.csv',index=False)
    slope=np.polyfit(np.log(group.N),np.log(group.mean_rank),1)[0]
    print(group[['N','mean_H','mean_C','mean_rank','mean_upper','mean_rank_over_sqrtN']].to_string(index=False,float_format=lambda z:f'{z:.3f}'))
    print('descriptive log slope',slope)
    (out/'graph_manifest.json').write_text(json.dumps(dict(seeds=args.seeds,sizes=ns,rng_rule='9715511+100003*N+seed',self_loops=False,signs='independent fair Rademacher',method='exact cyclotomic factor degree, no floating point rank',observed_loglog_slope=float(slope),description='post-hoc exploratory graph sampling; not publication confirmation'),indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.9))
    for c,name in [('mean_H','H (lower bound)'),('mean_rank','exact generic rank'),('mean_upper','H + C (upper bound)')]:
        axes[0].plot(group.N,group[c],marker='o',label=name)
    axes[0].set(xlabel='Number of neurons N',ylabel='Dimension',title='Signed functional-graph limit');axes[0].legend(fontsize=8)
    for c,name in [('mean_H_over_sqrtN','H'),('mean_rank_over_sqrtN','rank'),('mean_upper_over_sqrtN','H+C')]:
        axes[1].plot(group.N,group[c],marker='o',label=name)
    axes[1].set(xlabel='Number of neurons N',ylabel='Dimension / sqrt(N)',title='Square-root scaling diagnostic');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'graph_exact_scaling.png',dpi=190);fig.savefig(out/'graph_exact_scaling.pdf');plt.close(fig)

if __name__=='__main__':main()
