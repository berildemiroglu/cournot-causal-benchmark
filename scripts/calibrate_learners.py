import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cournot_sim import *
def nash(c,a=A_DEM,b=B_DEM):
    N=len(c); q=(a+c.sum()-(N+1)*c)/((N+1)*b); return b*q**2
N,nt=6,3
c=np.full(N,20.0); ct=c.copy(); ct[:nt]-=5
base=nash(c)[0]; p=nash(ct)
print("Nash TE=%.0f SP=%.0f GTE=%.0f"%(p[:nt].mean()-base,p[nt:].mean()-base, nash(np.full(N,15.0))[0]-base))
for gamma in [0.0,0.9]:
  for floor in [0.01,0.05,0.1]:
    for alpha in [0.05,0.2]:
        sim=simulate_worlds(300,N,0.0,nt,alpha,seed=1,hetero=False,gamma=gamma,floor=floor)
        tr=truths(sim)
        print(f"g={gamma} floor={floor} a={alpha} p0/Nash={sim['Y0'].mean()/base:.2f} conv={sim['conv0'].mean():.2f}",
              {k:round(v[0],1) for k,v in tr.items()})
