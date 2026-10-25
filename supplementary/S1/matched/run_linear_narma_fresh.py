#!/usr/bin/env python3
"""Held-out 30-lag linear-input baseline on the exact fresh NARMA10 seeds 400-419.
No reservoir state. Per-seed ridge selected ONLY on validation using stable
centered augmented least squares; train and test splits match existing comparison.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
SRC=Path(__file__).resolve().parent/'neurocomputing_2026'
sys.path.insert(0,str(SRC))
from frozen_nc_v2_1 import make_data
OUT=Path(__file__).resolve().parent
RIDGES=(0.,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1)
SEEDS=range(400,420);WASH=100;LENGTHS=(1200,500,500);LAG=30

def fitpredict(X,y,tr,vl,te,lam):
 A=X[tr];B=X[vl];C=X[te]
 mx=A.mean(0);my=y[tr].mean()
 At=A-mx;target=y[tr]-my
 if lam>0:
  a=np.concatenate([At,np.sqrt(lam)*np.eye(A.shape[1])],axis=0)
  z=np.concatenate([target,np.zeros(A.shape[1])])
 else:a,z=At,target
 coef=np.linalg.lstsq(a,z,rcond=1e-12)[0]
 pv=my+(B-mx)@coef;pt=my+(C-mx)@coef
 return np.sqrt(np.mean((pv-y[vl])**2)),np.sqrt(np.mean((pt-y[te])**2))

records=[]
for seed in SEEDS:
 U,y,sl=make_data('narma10',seed,LENGTHS,WASH)
 x=U[:,0]
 X=np.column_stack([np.roll(x,d) for d in range(LAG+1)])
 X[:LAG,:]=0.  # never used: first train index 100 > lag30
 risks=[fitpredict(X,y,*sl,lam) for lam in RIDGES]
 best=int(np.argmin([z[0] for z in risks]));validation,test=risks[best]
 records.append(dict(seed=seed,task='narma10',model='linear_30_lags',lag_start=0,lag_end=LAG,n_features=LAG+1,selected_ridge=RIDGES[best],validation_nrmse=validation,test_nrmse=test))
D=pd.DataFrame(records);D.to_csv(OUT/'linear_narma_fresh.csv',index=False)
# Original conventional outcomes for paired comparison
base=pd.read_csv(OUT/'evaluation_selected.csv').query("task=='narma10'").pivot(index='seed',columns='arm',values='selected_test_nrmse')
p=D.set_index('seed').join(base)
rng=np.random.default_rng(9727007)
rows=[]
for other in ['tcr','gaussian','cycle']:
 diff=p[other].to_numpy()-p.test_nrmse.to_numpy()
 draws=rng.choice(diff,size=(10000,len(diff)),replace=True).mean(1)
 rows.append(dict(contrast=other+'_minus_linear30',mean_delta=diff.mean(),ci95_low=np.quantile(draws,.025),ci95_high=np.quantile(draws,.975),n=len(diff)))
S=pd.DataFrame(rows);S.to_csv(OUT/'linear_narma_paired.csv',index=False)
(OUT/'linear_narma_design.json').write_text(json.dumps(dict(seeds=list(SEEDS),lag_inclusive=[0,30],n_features=31,split=LENGTHS,wash=WASH,standardization='frozen source make_data: train only',ridge_candidates=RIDGES,choice='per seed validation only',data_source='frozen_nc_v2_1.py unmodified',notes='not matched 80+13 calibration count, this is a transparent direct-input linear diagnostic; no state and no nonlinear features'),indent=2))
print('linear mean',D.test_nrmse.mean(),'std',D.test_nrmse.std());print(S.to_string(index=False))
