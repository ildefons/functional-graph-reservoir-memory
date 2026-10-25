#!/usr/bin/env python3
"""Paired s0={0.6,0.8} fixed-global-RMS memory sensitivity, seeds 500-519.
Includes lag 0 and two truncated SVD tolerances; original v2 source generation,
architecture signs, input sequences, and 20 seed IDs otherwise unchanged.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,time,sys
import numpy as np,pandas as pd
from scipy.linalg import svd
from scipy.special import softmax
from numba import njit
SCRIPT=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT))
from graph_exact_scaling import graph_stats
SCALES=(.6,.8); NS=(60,150); SEEDS=range(500,520)
TAUS=('zero',.08,.15,.53,3.5)
GAIN=.15; WASH=450;NTRAIN=3200;NTEST=1600
@njit(cache=True)
def simulate(W,b,u,gain,tanh):
 T=len(u);n=len(b);x=np.zeros(n);out=np.empty((T,n))
 for t in range(T):
  for i in range(n):
   z=gain*b[i]*u[t]
   for j in range(n):z+=W[i,j]*x[j]
   out[t,i]=np.tanh(z) if tanh else z
  for i in range(n):x[i]=out[t,i]
 return out

def evaluate(X,u,start,maxlag):
 delays=np.arange(maxlag+1)  # include current input as d=0
 itr=np.arange(start,start+NTRAIN);ite=np.arange(start+NTRAIN,start+NTRAIN+NTEST)
 Ytr=u[itr[:,None]-delays];Yte=u[ite[:,None]-delays]
 A=X[itr];C=X[ite];mx=A.mean(0);my=Ytr.mean(0);A=A-mx;C=C-mx
 U,s,Vt=svd(A,full_matrices=False,check_finite=False,lapack_driver='gesdd')
 results={}
 for rtol in [1e-9,1e-11]:
  keep=s>max(s[0]*rtol,1e-12)
  B=(Vt[keep].T /s[keep])@(U[:,keep].T@(Ytr-my))
  pred=C@B+my
  pc=pred-pred.mean(0);yc=Yte-Yte.mean(0)
  cor=np.sum(pc*yc,axis=0)**2/np.maximum(np.sum(pc**2,axis=0)*np.sum(yc**2,axis=0),1e-30)
  r2=1-np.mean((pred-Yte)**2,axis=0)/np.var(Yte,axis=0)
  results[rtol]=(cor,r2,int(keep.sum()))
 return results

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(SCRIPT));args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 rows=[];curves=[];t0=time.time()
 for n in NS:
  maxlag=min(2*n,180)
  for seed in SEEDS:
   rng=np.random.default_rng(990721+n*10000+seed)
   a=rng.normal(size=(n,n));np.fill_diagonal(a,-np.inf)
   sg=rng.choice([-1.,1.],size=(n,n));np.fill_diagonal(sg,0)
   b=rng.uniform(-1,1,size=n)
   u=rng.uniform(-1,1,size=WASH+maxlag+NTRAIN+NTEST+1)
   P=np.zeros((n,n));succ=np.argmax(a,axis=1);P[np.arange(n),succ]=sg[np.arange(n),succ]
   gs=graph_stats(succ,sg[np.arange(n),succ]);bound=gs['generic_krylov_rank_exact']
   raw={'tau=zero':P}
   for tau in TAUS[1:]:raw[f'tau={tau}']=sg*softmax(a/tau,axis=1)
   rng2=np.random.default_rng(990721+n*10000+seed+999999)
   gaus=rng2.normal(size=(n,n));np.fill_diagonal(gaus,0);raw['gaussian']=gaus
   cycle=np.zeros((n,n));cycle[np.arange(n),(np.arange(n)+1)%n]=rng2.choice([-1.,1.],size=n);raw['cycle']=cycle
   for arm,A in raw.items():
    norm=A/(np.sqrt(n)*np.sqrt(np.mean(A*A)))
    rho_unit=float(np.max(np.abs(np.linalg.eigvals(norm))))
    for s0 in SCALES:
     W=np.ascontiguousarray(norm*s0)
     for nonlinear in [False,True]:
      X=simulate(W,b,u,GAIN,nonlinear)
      if not np.isfinite(X).all():raise ValueError((n,seed,arm,s0,nonlinear,'diverged'))
      cases=evaluate(X,u,WASH+maxlag,maxlag)
      for rtol,(caps,r2,rank) in cases.items():
       rows.append(dict(N=n,seed=seed,arm=arm,s0=s0,dynamics='tanh' if nonlinear else 'linear',svd_rtol=rtol,
        mc_lag0=float(caps[0]),mc_positive=float(caps[1:].sum()),mc_total=float(caps.sum()),mc_r2_total=float(r2.sum()),
        delay10_corr2=float(caps[10]),numerical_rank=rank,graph_exact_degree=bound if arm=='tau=zero' else np.nan,
        spectral_radius=s0*rho_unit,input_gain=GAIN))
       if rtol==1e-9:
        for lag,cap in enumerate(caps):curves.append((n,seed,arm,s0,'tanh' if nonlinear else 'linear',lag,float(cap),float(r2[lag])))
   print(f'completed N={n} seed={seed} elapsed={time.time()-t0:.2f}s',flush=True)
 d=pd.DataFrame(rows);d.to_csv(out/'memory_sensitivity_per_seed.csv',index=False)
 pd.DataFrame(curves,columns=['N','seed','arm','s0','dynamics','lag','memory_corr2','memory_r2']).to_csv(out/'memory_sensitivity_by_delay.csv',index=False)
 summary=d.groupby(['N','s0','dynamics','arm','svd_rtol'],as_index=False).agg(mc_lag0=('mc_lag0','mean'),mc_positive=('mc_positive','mean'),mc_total=('mc_total','mean'),mc_sd=('mc_total','std'),delay10=('delay10_corr2','mean'),num_rank=('numerical_rank','mean'),graph_degree=('graph_exact_degree','mean'),mean_spectral=('spectral_radius','mean'))
 summary.to_csv(out/'memory_sensitivity_summary.csv',index=False)
 (out/'design_v3.json').write_text(json.dumps(dict(N=NS,seeds=list(SEEDS),s0=SCALES,svd_rtol=[1e-9,1e-11],gain=GAIN,nsamples=[WASH,NTRAIN,NTEST],maxlag='min(2*N,180)',zero_lag_included=True,architecture='same frozen scores/signs and inputs as v2 500-519; all arms same RMS normalization',notes=['no noise','unregularized truncated SVD','no retuning','N=150 lagmax=180 > N'],),indent=2))
 print(summary[(summary.svd_rtol==1e-9)&(summary.dynamics=='linear')].to_string(index=False),flush=True)
if __name__=='__main__':main()
