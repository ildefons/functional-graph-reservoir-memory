import numpy as np, scipy.linalg as la, csv, time
from pathlib import Path
out=Path(__file__).resolve().parent; N=60; s0=.8; seeds=20; taus=[0,.02,.08,.15,.53,3.5]; noises=[1e-6,1e-4,1e-2,1e-1]
rows=[]
def make_w(scores,signs,tau):
 if tau==0:
  p=np.zeros_like(scores); p[np.arange(N),np.argmax(scores,axis=1)]=1
 else:
  x=scores/tau; x=x-np.max(x,axis=1,keepdims=True); p=np.exp(x); p[np.arange(N),np.arange(N)]=0;p/=p.sum(axis=1,keepdims=True)
 m=p*signs;q=np.sum(m*m)/N;return s0*m/np.sqrt(q)
def measurements(W,b,eps):
 # Solve two Lyapunov equations via Schur stability, input signal and state process noise
 G=np.outer(b,b)
 S=la.solve_discrete_lyapunov(W,G)
 U=la.solve_discrete_lyapunov(W,np.eye(N))
 C=S+eps*U
 # use symmetrization and cholesky; cap bounded by rank if conditioning extreme
 C=(C+C.T)/2
 L=la.cho_factor(C,check_finite=False)
 total=np.trace(la.cho_solve(L,S,check_finite=False)).real
 # individual delays
 v=b.copy();delay={}
 for k in range(61):
  if k in [0,1,5,10,20,40,60]:delay[k]=float(v @ la.cho_solve(L,v,check_finite=False))
  v=W@v
 return float(total),delay
start=time.time()
for seed in range(seeds):
 rng=np.random.default_rng(6000+seed)
 scores=rng.normal(size=(N,N));np.fill_diagonal(scores,-np.inf)
 signs=rng.choice([-1.,1.],size=(N,N));np.fill_diagonal(signs,0)
 b=rng.uniform(-1,1,size=N); b=b/np.linalg.norm(b)
 for typ in ['temperature','gaussian','cycle']:
  params=taus if typ=='temperature' else ['baseline']
  for par in params:
   if typ=='temperature': W=make_w(scores,signs,par)
   elif typ=='gaussian':
    r=rng.normal(size=(N,N));np.fill_diagonal(r,0);W=r*(s0*np.sqrt(N)/la.norm(r,'fro'))
   else:
    W=np.zeros((N,N));W[np.arange(N),(np.arange(N)+1)%N]=s0*rng.choice([-1,1],size=N)
   rho=max(abs(la.eigvals(W)))
   if rho>=1:
    rows.append(dict(seed=seed,arm=typ,tau=par,rho=rho,eps=np.nan,total=np.nan,skip='unstable'));continue
   for eps in noises:
    total,delay=measurements(W,b,eps)
    rows.append(dict(seed=seed,arm=typ,tau=par,rho=float(rho),eps=eps,total=total,**{'m'+str(k):v for k,v in delay.items()},skip=''))
 print('seed',seed,'time',round(time.time()-start,1),flush=True)
with (out/'tcr_noise_pilot_results.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('FINISHED',len(rows),'runtime',round(time.time()-start,1))
