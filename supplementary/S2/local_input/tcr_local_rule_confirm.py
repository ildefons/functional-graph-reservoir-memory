import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np, scipy.linalg as la, csv, time
from pathlib import Path
OUT=Path(__file__).resolve().parent;OUT.mkdir(parents=True,exist_ok=True)
TAUS=[0.,.08,.53,3.5]; EPS=[1e-5,1e-3,1e-2]
def cycles_from_f(f):
 n=len(f);done=np.zeros(n,bool);cycles=[]
 for start in range(n):
  if done[start]:continue
  local={};seq=[];v=start
  while not done[v] and v not in local:
   local[v]=len(seq);seq.append(v);v=f[v]
  if v in local: cycles.append(seq[local[v]:])
  done[seq]=True
 return cycles

def local_dims(f):
 n=len(f);children=[[] for _ in range(n)]
 for i,parent in enumerate(f):children[parent].append(i)
 cyc=cycles_from_f(f);dims=np.empty(n,int)
 # distance via shortest-path BFS in signal-flow graph
 for j in range(n):
  dist=np.full(n,-1,int);dist[j]=0;q=[j]
  for u in q:
   for v in children[u]:
    if dist[v]<0:dist[v]=dist[u]+1;q.append(v)
  dims[j]=max(dist)+1
 return dims,cyc

def make_w(scores,signs,tau,s0):
 n=len(scores)
 if tau==0:
  p=np.zeros_like(scores);p[np.arange(n),np.argmax(scores,axis=1)]=1
 else:
  z=scores/tau;z-=np.max(z,axis=1,keepdims=True)
  p=np.exp(z);np.fill_diagonal(p,0);p/=p.sum(axis=1,keepdims=True)
 return s0*signs*p / np.sqrt(np.sum(p*p)/n)

def mc_scores(w,L,vecs,eps,horizon):
 # vecs B (n,k), propagate all candidate columns together (n,k) rather than repeated loops
 n,k=vecs.shape;hist=np.empty((n,k,horizon));v=vecs.copy()
 for h in range(horizon): hist[:,:,h]=v;v=w@v
 y=la.solve_triangular(L,hist.reshape(n,-1),lower=True,check_finite=False).reshape(n,k,horizon)
 # MC equals tr G(G+eps I)^-1 via eigenvalues of Y Y^T (whitened)
 ans=[]
 for c in range(k):
  ev=la.eigvalsh(y[:,c,:].T@y[:,c,:],check_finite=False);ev=np.maximum(ev,0)
  ans.append([float(np.sum(ev/(ev+e))) for e in eps])
 return np.array(ans),float(np.max(la.norm(v,axis=0)))

# proof audit numeric for small random f
rng=np.random.default_rng(90217);audits=[]
for n in [6,9,14]:
 for z in range(10):
  s=rng.normal(size=(n,n));np.fill_diagonal(s,-np.inf)
  f=np.argmax(s,axis=1);sign=rng.choice([-1.,1.],size=(n,n));np.fill_diagonal(sign,0)
  W=make_w(s,sign,0.,.8);dims,_=local_dims(f)
  for j in range(n):
   k=np.empty((n,n));v=np.eye(n)[:,j]
   for t in range(n):k[:,t]=v;v=W@v
   rank=np.linalg.matrix_rank(k,tol=1e-8)
   audits.append((n,z,j,int(dims[j]),int(rank)))
assert all(a[-2]==a[-1] for a in audits), [a for a in audits if a[-2]!=a[-1]][:10]
print('THEOREM AUDIT',len(audits),'matches',flush=True)

records=[];run_start=time.time()
for n,count in [(150,12),(500,10)]:
 for s0 in [.6,.8]:
  for seed in range(count):
   rng=np.random.default_rng(5309000+n*300+int(s0*100)*1000+seed)
   a=rng.normal(size=(n,n));np.fill_diagonal(a,-np.inf)
   signs=rng.choice([-1.,1.],size=(n,n));np.fill_diagonal(signs,0)
   f=np.argmax(a,axis=1);dims,cyc=local_dims(f)
   allcyc=set(sum(cyc,[]));longest=sorted(cyc,key=lambda c:(-len(c),min(c)))[0]
   # fixed deterministic tie handling picks lowest index
   jstar=int(np.flatnonzero(dims==dims.max())[0]);rsel=np.random.default_rng(870011+n*100+seed+int(s0*100)*100)
   randomnodes=rsel.choice(n,size=13,replace=False).tolist()
   dense=rsel.normal(size=(n,13));dense/=la.norm(dense,axis=0)
   # all nodes for N150; at N500 evaluate all single-neuron candidates too when feasible
   nodes=list(range(n))
   B=np.zeros((n,n+13));B[np.arange(n),np.arange(n)]=1;B[:,n:]=dense
   for tau in TAUS:
    W=make_w(a,signs,tau,s0);rho=float(np.max(np.abs(la.eigvals(W,check_finite=False))))
    if rho>=1:
     for ep in EPS: records.append(dict(n=n,s0=s0,seed=seed,tau=tau,eps=ep,status='unstable',rho=rho))
     continue
    U=la.solve_discrete_lyapunov(W,np.eye(n));L=la.cholesky((U+U.T)/2,lower=True,check_finite=False)
    horizon=140 if n==150 else 160
    vals,tail=mc_scores(W,L,B,EPS,horizon)
    for ei,ep in enumerate(EPS):
     v=vals[:,ei];best=int(np.argmax(v[:n]));lb=max(longest,key=lambda j:v[j]);rb=max(randomnodes,key=lambda j:v[j]);db=n+int(np.argmax(v[n:]));
     records.append(dict(n=n,s0=s0,seed=seed,tau=tau,eps=ep,status='ok',rho=rho,
      local_max_dim=int(dims[jstar]),local_rule_node=jstar,local_rule_on_cycle=int(jstar in allcyc),
      max_node=best,max_node_dim=int(dims[best]),max_node_on_cycle=int(best in allcyc),
      rule=float(v[jstar]),oracle=float(v[best]),longest_best=float(v[lb]),random13_best=float(v[rb]),
      dense13_best=float(v[db]),random1=float(v[randomnodes[0]]),
      regret=float(v[best]-v[jstar]),adv_random13=float(v[jstar]-v[rb]),adv_dense13=float(v[jstar]-v[db]),
      ncyclic=len(allcyc),longest_len=len(longest),tail=tail))
   print('finished',n,s0,seed+1,'/',count,'elapsed',round(time.time()-run_start,1),flush=True)
with (OUT/'results.csv').open('w',newline='') as g:
 wr=csv.DictWriter(g,fieldnames=sorted(set().union(*(r.keys() for r in records))));wr.writeheader();wr.writerows(records)
with (OUT/'rank_audit.csv').open('w',newline='') as g:
 wr=csv.writer(g);wr.writerow(['n','seed','node','predicted','rank']);wr.writerows(audits)
print('COMPLETE',len(records),'rows seconds',round(time.time()-run_start,1),flush=True)
