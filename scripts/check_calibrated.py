"""Check the chosen learner setting across market sizes (heterogeneous costs, one seed, diagnostic only)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from cournot_sim import simulate_worlds, truths
SETTING = dict(gamma=0.0, floor=0.03, T1=20000)   # post period = 17,500 periods
for N in [4, 8, 16]:
    for name, alpha in [("slow", 0.05), ("fast", 0.10)]:
        sim = simulate_worlds(300, N, 0.0, N // 2, alpha, seed=3, hetero=True, **SETTING)
        tr = truths(sim)
        base = (80.0 ** 2) / (N + 1) ** 2
        print(f"N={N:2d} {name} conv={sim['conv0'].mean():.2f} p0/Nash={sim['Y0'].mean()/base:.2f} ",
              {k: round(v[0], 1) for k, v in tr.items()})
