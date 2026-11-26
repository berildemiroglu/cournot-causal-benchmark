"""Learner calibration grid: where does feedback (spillover ~ Nash) appear, and how much converges?
Setting: N=6, 3 treated, homogeneous costs (Nash TE=73, SP=-44, GTE=17). One seed, diagnostic only."""
import os, sys, itertools, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from cournot_sim import simulate_worlds, truths

rows = []
t0 = time.time()
for T1, floor, alpha, gamma in itertools.product([5000, 10000, 20000], [0.01, 0.03, 0.05], [0.02, 0.05, 0.1, 0.2], [0.0, 0.9]):
    sim = simulate_worlds(300, 6, 0.0, 3, alpha, seed=1, hetero=False, gamma=gamma, floor=floor, T1=T1)
    tr = truths(sim)
    rows.append(dict(post_len=T1 - 2500, floor=floor, alpha=alpha, gamma=gamma,
                     conv=sim['conv0'].mean(), profit_vs_nash=sim['Y0'].mean() / (80.0**2 / 49.0),
                     TE=tr['TE'][0], SP=tr['SP'][0], GTE=tr['GTE'][0]))
df = pd.DataFrame(rows).round(2)
os.makedirs("results", exist_ok=True)
df.to_csv("results/calibration_grid.csv", index=False)
print(df.to_string(index=False)); print("time", round(time.time() - t0))
