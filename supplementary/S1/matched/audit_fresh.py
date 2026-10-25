import numpy as np,pandas as pd,sys,json
from pathlib import Path
from scipy.linalg import lstsq
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_fresh_matched import make_data,build_bases,tcr_matrices,FACTORS,drive_bank,LENGTHS,WASH,N
p=Path(__file__).resolve().parent; sel=pd.read_csv(p/'evaluation_selected.csv');cal=pd.read_csv(p/'calibration_selected.csv'); rows=[]
for task in ('memory_d10','narma10','lorenz_x'):
    pars={r.arm:r for r in cal[cal.task==task].itertuples()}
    for seed in range(400,420):
        U,y,sl=make_data(task,seed,LENGTHS,WASH);bases,win=build_bases(task,seed)
        for arm in ('tcr','gaussian','cycle'):
            pr=pars[arm];r=sel[(sel.task==task)&(sel.seed==seed)&(sel.arm==arm)].iloc[0]
            ix=int(r.selected_index)
            W=tcr_matrices(task,seed,pr.s0)[ix] if arm=='tcr' else bases[arm]*pr.s0*FACTORS[ix]
            X=drive_bank(np.ascontiguousarray(W[None,:,:]),np.ascontiguousarray(win*pr.input_scale),np.ascontiguousarray(U[:,0]))[0]
            Xtr=X[sl[0]];Xte=X[sl[2]];xm=Xtr.mean(0);ym=y[sl[0]].mean()
            A=Xtr-xm;Y=y[sl[0]]-ym
            ridge=float(pr.ridge)
            aug=np.vstack((A,np.sqrt(ridge)*np.eye(N)))
            goal=np.concatenate([Y,np.zeros(N)])
            coef,_,_,_=lstsq(aug,goal,lapack_driver='gelsd')
            pred=(Xte-xm)@coef+ym
            stable=np.sqrt(np.mean((pred-y[sl[2]])**2));recorded=r.selected_test_nrmse
            c=np.linalg.cond(A)
            rows.append(dict(task=task,seed=seed,arm=arm,recorded=recorded,stable=stable,abs_difference=abs(stable-recorded),centered_design_cond=c,lambda_=ridge))
d=pd.DataFrame(rows);d.to_csv(p/'selected_solver_audit.csv',index=False)
print('n cases',len(d))
print('maximum NRMSE diff',d.abs_difference.max())
print('mean NRMSE diff',d.abs_difference.mean())
print('cases > 1e-6',int((d.abs_difference>1e-6).sum()))
print('worst cases')
print(d.nlargest(10,'abs_difference')[['task','seed','arm','recorded','stable','abs_difference','centered_design_cond','lambda_']].to_string(index=False,float_format=lambda x:f'{x:.6g}'))
