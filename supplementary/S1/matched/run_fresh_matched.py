#!/usr/bin/env python3
"""Secondary, post-hoc, matched-setup comparison against canonical ESN and simple cycle.

Reuses frozen reference data/substrates and splits, but recalibrates ALL arms
using exactly the same expanded 80-point s0/input/ridge budget on seeds 70,71.
Then ALL arms get 13 validation-selected architecture candidates on each of the
evaluation seeds 400:419. These seeds are NOT independent confirmation
of this post-hoc analysis. Original tagged experiments remain unchanged.

Three arms: TCR row-softmax RMS (13 tau), dense Gaussian RMS (13 recurrent gains),
signed simple cycle (13 gains). s0 is the recurrent RMS target at ref/factor1;
all arms calibrate their own s0/input gain/ridge at the same reference setting.
"""
import os;os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,json,time,sys
from pathlib import Path
from numba import njit
from scipy.linalg import eigh
source=Path(__file__).parent/'neurocomputing_2026'
sys.path.insert(0,str(source))
from frozen_nc_v2_1 import make_data,make_base,norm_rms,seed_for

ARMS=['tcr','gaussian','cycle']
TASKS=['memory_d10','narma10','lorenz_x']
TAUS=np.geomspace(.08,3.5,13)
FACTORS=np.geomspace(0.35,1/0.35,13)
CAL_S0=(.2,.5,.8,1.2)
CAL_IN=(.002,.01,.05,.2,.8)
CAL_RIDGE=(1e-11,1e-9,1e-7,1e-5)
LENGTHS=(1200,500,500); WASH=100;N=60
CAL_SEEDS=(70,71); EVAL_SEEDS=tuple(range(400,420))

@njit(cache=True)
def drive(W,win,u):
    T=u.shape[0];n=W.shape[0]; x=np.zeros(n);out=np.empty((T,n))
    for t in range(T):
        v=win*u[t]
        for i in range(n):
            z=v[i]
            for j in range(n):z+=W[i,j]*x[j]
            out[t,i]=np.tanh(z)
        for i in range(n):x[i]=out[t,i]
    return out

@njit(cache=True)
def drive_bank(W,win,u):
    K,n=W.shape[:2];T=len(u);x=np.zeros((K,n));out=np.empty((K,T,n))
    for t in range(T):
        for k in range(K):
            for i in range(n):
                z=win[i]*u[t]
                for j in range(n):z+=W[k,i,j]*x[k,j]
                out[k,t,i]=np.tanh(z)
        for k in range(K):
            for i in range(n):x[k,i]=out[k,t,i]
    return out

def build_bases(task,seed):
    scores,signs,win=make_base(N,task,seed)
    logits=scores/TAUS[len(TAUS)//2]
    exp=np.exp(logits-np.max(logits,axis=1,keepdims=True))
    alloc=signs*exp/np.sum(exp,axis=1,keepdims=True)
    gaussian=scores.copy();np.fill_diagonal(gaussian,0)
    ring=np.zeros((N,N))
    tgt=np.arange(N);src=(tgt+1)%N
    ring[tgt,src]=signs[tgt,src]
    return {'tcr':norm_rms(alloc,1.),'gaussian':norm_rms(gaussian,1.),'cycle':norm_rms(ring,1.)}, win[:,0]

def tcr_matrices(task,seed,s0):
    scores,signs,win=make_base(N,task,seed)
    a=[]
    for tau in TAUS:
        l=scores/tau;p=np.exp(l-np.max(l,axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
        a.append(norm_rms(signs*p,s0))
    return np.array(a)

def regression_preds(X,y,train,val,test,ridges):
    # QR-equivalent eigen solution on centered X (unpenalized intercept).
    A=X[train];yv=y[train];xm=A.mean(axis=0);ym=yv.mean();A=A-xm; yv=yv-ym
    G=A.T@A;diag,v=eigh(G,check_finite=False);diag=np.maximum(diag,0.)
    q=v.T@(A.T@yv)
    vproj=(X[val]-xm)@v
    tproj=(X[test]-xm)@v if test is not None else None
    target_val=y[val];target_test=y[test] if test is not None else None
    out=[]
    for lam in ridges:
        weight=q/(diag+lam)
        pred_val=ym+vproj@weight
        nval=float(np.sqrt(np.mean((pred_val-target_val)**2)))
        if test is None:out.append((nval,None))
        else:
            pred_test=ym+tproj@weight
            out.append((nval,float(np.sqrt(np.mean((pred_test-target_test)**2)))))
    return out

def calibrated(out):
    records=[]
    start=time.time()
    for task in TASKS:
        for seed in CAL_SEEDS:
            U,y,sl=make_data(task,seed,LENGTHS,WASH)
            base,win=build_bases(task,seed)
            for arm in ARMS:
                for s0 in CAL_S0:
                    W=base[arm]*s0
                    for inp in CAL_IN:
                        X=drive(W,win*inp,U[:,0])
                        risks=regression_preds(X,y,sl[0],sl[1],None,CAL_RIDGE)
                        for ridge,(val,_) in zip(CAL_RIDGE,risks):
                            records.append(dict(task=task,seed=seed,arm=arm,s0=s0,input_scale=inp,ridge=ridge,val_nrmse=val))
            print('calibration',task,seed,'elapsed',round(time.time()-start,1),flush=True)
    d=pd.DataFrame(records)
    d.to_csv(out/'calibration_full.csv',index=False)
    g=d.groupby(['task','arm','s0','input_scale','ridge'],as_index=False).val_nrmse.mean()
    h=g.sort_values(['val_nrmse','s0','input_scale','ridge']).groupby(['task','arm'],as_index=False).first()
    h.to_csv(out/'calibration_selected.csv',index=False)
    print(h.to_string(index=False),flush=True)
    return h

def evaluate(out,h):
    records=[];start=time.time()
    for task in TASKS:
        params={r.arm:r for r in h[h.task==task].itertuples()}
        for seed in EVAL_SEEDS:
            U,y,sl=make_data(task,seed,LENGTHS,WASH)
            bases,win=build_bases(task,seed)
            for arm in ARMS:
                pr=params[arm]
                if arm=='tcr': mats=tcr_matrices(task,seed,pr.s0)
                else:mats=np.stack([bases[arm]*pr.s0*factor for factor in FACTORS])
                trajectories=drive_bank(np.ascontiguousarray(mats),win*pr.input_scale,np.ascontiguousarray(U[:,0]))
                for j in range(len(TAUS)):
                    v,t=regression_preds(trajectories[j],y,sl[0],sl[1],sl[2],[pr.ridge])[0]
                    records.append(dict(task=task,seed=seed,arm=arm,index=j,tau=float(TAUS[j]) if arm=='tcr' else np.nan,gain_factor=float(FACTORS[j]) if arm!='tcr' else np.nan,
                                        val_nrmse=v,test_nrmse=t,s0=float(pr.s0),input_scale=float(pr.input_scale),ridge=float(pr.ridge)))
            print('evaluation',task,seed,'elapsed',round(time.time()-start,1),flush=True)
    df=pd.DataFrame(records);df.to_csv(out/'evaluation_all_candidates.csv',index=False)
    rows=[]
    for (task,seed,arm),g in df.groupby(['task','seed','arm']):
        selected=g.loc[g.val_nrmse.idxmin()]
        rows.append(dict(task=task,seed=seed,arm=arm,selected_index=int(selected['index']),selected_val_nrmse=float(selected.val_nrmse),selected_test_nrmse=float(selected.test_nrmse),mean_grid_test_nrmse=float(g.test_nrmse.mean())))
    selected=pd.DataFrame(rows);selected.to_csv(out/'evaluation_selected.csv',index=False)
    return selected

def paired_bootstrap(v,out):
    rng=np.random.default_rng(493993)
    stats=[]
    for task in TASKS:
        table=v[v.task==task].pivot(index='seed',columns='arm',values='selected_test_nrmse')
        for arm in ('gaussian','cycle'):
            dif=(table['tcr']-table[arm]).to_numpy();n=len(dif)
            idx=rng.integers(0,n,size=(5000,n));means=dif[idx].mean(1)
            lo,hi=np.quantile(means,[.025,.975]);stats.append(dict(task=task,contrast='tcr_minus_'+arm,n=n,mean_difference=float(dif.mean()),ci_low=float(lo),ci_high=float(hi),wins=int(np.sum(dif<0))))
    outdf=pd.DataFrame(stats);outdf.to_csv(out/'paired_statistics.csv',index=False)
    print('SELECTION MEANS',flush=True)
    print(v.groupby(['task','arm']).selected_test_nrmse.mean().unstack().to_string(),flush=True)
    print('PAIRED',outdf.to_string(index=False),flush=True)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='matched');parser.add_argument('--stage',choices=['calibrate','evaluate','all'],default='all');args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'design.json').write_text(json.dumps(dict(description='fresh seed evaluation 400-419 under previously completed matched calibration 70-71; designs informed by previous inspection, not prospective preregistration',N=N,cal_seeds=CAL_SEEDS,eval_seeds=EVAL_SEEDS,tasks=TASKS,taus=TAUS.tolist(),baseline_factors=FACTORS.tolist(),s0_candidates=CAL_S0,input_candidates=CAL_IN,ridge_candidates=CAL_RIDGE,calibration_budget_per_arm_per_task=len(CAL_S0)*len(CAL_IN)*len(CAL_RIDGE),evaluation_arch_candidates=13,protocol='released frozen_nc_v2_1 data/substrate generator, train/validation/test unchanged',pre_registered=False),indent=2))
    if args.stage in ('calibrate','all'):h=calibrated(out)
    else:h=pd.read_csv(out/'calibration_selected.csv')
    if args.stage in ('evaluate','all'):
        v=evaluate(out,h);paired_bootstrap(v,out)
if __name__=='__main__':main()
