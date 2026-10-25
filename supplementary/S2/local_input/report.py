import numpy as np,pandas as pd
from scipy import stats
from pathlib import Path
p=Path(__file__).parent
x=pd.read_csv(p/'results.csv');v=x[x.status=='ok'].copy()
rows=[]
for keys,t in v.groupby(['n','s0','eps','tau']):
 row=dict(zip(['n','s0','eps','tau'],keys));row['seeds']=len(t)
 for var in ['rule','oracle','regret','adv_random13','adv_dense13']:
  a=t[var].to_numpy();mean=a.mean();se=stats.sem(a) if len(a)>1 else 0
  low,high=stats.t.interval(.95,len(a)-1,loc=mean,scale=se)
  row[var+'_mean']=mean;row[var+'_ci_low']=low;row[var+'_ci_high']=high
 rows.append(row)
pd.DataFrame(rows).to_csv(p/'summary.csv',index=False)
print(pd.DataFrame(rows).query('eps == 0.001')[['n','s0','tau','seeds','rule_mean','oracle_mean','regret_mean','adv_random13_mean','adv_random13_ci_low','adv_random13_ci_high','adv_dense13_mean']].round(2).to_string(index=False))
print('unstable counts',x[x.status=='unstable'].groupby(['n','s0','tau']).size().to_dict())
