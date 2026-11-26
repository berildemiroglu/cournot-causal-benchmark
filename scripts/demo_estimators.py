"""Demo: run all estimators on one simulated data set, fixed vs varying treated share.
Truths (DE, TE, SP, GTE) come from the paired worlds. One seed, diagnostic only."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from cournot_sim import simulate_worlds, truths
from estimators import make_data, run_all

SETTING = dict(gamma=0.0, floor=0.03, T1=20000)    # calibrated learner setting (see calibrate_grid.py)
N, M, ALPHA, KAPPA = 6, 400, 0.05, 1.0
rng = np.random.default_rng(0)

for label, n_treat in [("fixed share 0.5", 3), ("share varies 1/3/5", rng.choice([1, 3, 5], size=M))]:
    sim = simulate_worlds(M, N, KAPPA, n_treat, ALPHA, seed=11, hetero=True, **SETTING)
    tr = truths(sim)
    print(f"\n=== {label}: N={N}, M={M}, kappa={KAPPA}, alpha={ALPHA}")
    print("truth:", {k: f"{v[0]:.1f} (mc se {v[1]:.1f})" for k, v in tr.items()}, " TE-SP = %.1f" % (tr['TE'][0] - tr['SP'][0]))
    t = time.time()
    res = run_all(make_data(sim))
    for name, r in res.items():
        if 'error' in r:
            print(f"  {name:18s} ERROR {r['error']}")
            continue
        print(f"  {name:18s}", "  ".join(f"{k}={v[0]:8.1f} ({v[1]:5.1f})" for k, v in r.items()))
    print(f"  [{time.time()-t:.0f}s]")
