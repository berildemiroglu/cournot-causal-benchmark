import time, numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cournot_sim import *

def nash_profit(c, a=A_DEM, b=B_DEM):
    N=len(c); q=(a+c.sum()-(N+1)*c)/((N+1)*b); return b*q**2

t=time.time()
for N,nt in [(3,1),(6,3),(12,6)]:
    sim=simulate_worlds(300,N,0.0,nt,0.1,seed=1,hetero=False)
    tr=truths(sim)
    c=np.full(N,20.0); ct=c.copy(); ct[:nt]-=5
    base=nash_profit(c)[0]; pt=nash_profit(ct)
    print(f"N={N} nt={nt} t={time.time()-t:.1f}s  profit0/Nash={sim['Y0'].mean()/base:.2f}  conv={sim['conv0'].mean():.2f}")
    print("  sim  ", {k:(round(v[0],1),round(v[1],1)) for k,v in tr.items()})
    print("  Nash  TE=%.0f SP=%.0f"%(pt[:nt].mean()-base, pt[nt:].mean()-base))
# zero-effect / placebo check: worlds identical => exactly zero
sim=simulate_worlds(200,6,1.0,3,0.1,seed=2,placebo=True)
print("placebo:",{k:round(v[0],6) for k,v in truths(sim).items()})
# CRN check: reproducibility
a=simulate_worlds(100,6,1.0,3,0.1,seed=5); b=simulate_worlds(100,6,1.0,3,0.1,seed=5)
print("reproducible:",np.array_equal(a['Yobs'],b['Yobs']))
