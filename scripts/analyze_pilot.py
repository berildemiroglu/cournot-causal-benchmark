"""Pilot analysis: bias / RMSE / coverage per estimator x estimand x cell, placebo rejection,
convergence diagnostics and the go/no-go checks (a)-(d) of the proposal.

Usage: python scripts/analyze_pilot.py results/pilot_raw.csv
Thresholds below are PLACEHOLDERS (the proposal fixes them after the pilot, before the main run).
"""
import sys
import numpy as np
import pandas as pd

TRUTH_SE_FRAC = 0.10     # (a) MC s.e. of each truth below this share of |truth|
PLACEBO_RANGE = (0.02, 0.10)   # (b) rejection rate near nominal 5%
CONV_MIN = 0.50          # (c) share of markets converging, per cell
Z = 1.96

path = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_raw.csv"
df = pd.read_csv(path)
df = df[df.est.notna() & (df.quantity != "ERROR")]
errors = pd.read_csv(path)
n_err = (errors.quantity == "ERROR").sum()
if n_err:
    print(f"WARNING: {n_err} estimator runs failed (rows with quantity=ERROR)\n")
pol = df[df.placebo == 0].copy()
pla = df[df.placebo == 1].copy()

# -- truths per cell: mean over replications, with MC s.e. across replications
rep_truth = (pol.drop_duplicates(["cell", "rep"])[["cell", "N", "design", "kappa", "speed", "rep",
                                                    "true_DE", "true_TE", "true_SP", "true_GTE"]])
rep_truth["true_TESP"] = rep_truth.true_TE - rep_truth.true_SP
tcols = ["true_DE", "true_TE", "true_SP", "true_GTE", "true_TESP"]
g = rep_truth.groupby("cell")
truth = g[tcols].mean()
truth_se = g[tcols].std() / np.sqrt(g.size().values[:, None])
print("=== (a) Truth stability: MC s.e. / |truth| per cell (should be < %.2f)" % TRUTH_SE_FRAC)
ratio = (truth_se / truth.abs()).round(3)
print(ratio.to_string())
print("-> pass" if (ratio.max().max() < TRUTH_SE_FRAC) else "-> FAIL in some cells (more repetitions / markets needed)")

# -- accuracy metrics against each estimand
targets = {"DE": "true_DE", "TE": "true_TE", "TE-SP": "true_TESP", "GTE": "true_GTE"}
pol = pol.join(truth[tcols].rename(columns=lambda c: c + "_cell"), on="cell")
pol["true_TESP_cell"] = pol.true_TE_cell - pol.true_SP_cell
rows = []
for (cell, est, q), d in pol.groupby(["cell", "estimator", "quantity"]):
    for tname, tcol in targets.items():
        tv = d[tcol + "_cell"].iloc[0]
        err = d.est - tv
        cover = ((d.est - Z * d.se <= tv) & (tv <= d.est + Z * d.se)).astype(float)
        rows.append(dict(cell=cell, estimator=est, quantity=q, target=tname, truth=tv, n=len(d),
                         bias=err.mean(), bias_mcse=err.std() / np.sqrt(len(d)),
                         rmse=np.sqrt((err ** 2).mean()), coverage=cover.mean(),
                         rel_bias=err.mean() / tv if tv != 0 else np.nan))
acc = pd.DataFrame(rows)
acc.to_csv(path.replace("_raw.csv", "_accuracy.csv"), index=False)

print("\n=== Accuracy: relative bias of tau_D (own-treatment coefficient) against each estimand")
tau = acc[acc.quantity == "tau_D"]
piv = tau.pivot_table(index=["cell", "estimator"], columns="target", values="rel_bias").round(2)
print(piv.to_string())

print("\n=== Exposure-aware GTE estimates (share-varying designs): relative bias and coverage against GTE")
gte = acc[(acc.quantity == "GTE") & (acc.target == "GTE")][["cell", "estimator", "truth", "bias", "bias_mcse", "rel_bias", "rmse", "coverage"]]
print(gte.round(3).to_string(index=False))

# -- (b) placebo
pla["reject"] = (pla.est.abs() / pla.se > Z).astype(float)
pr = pla[pla.quantity == "tau_D"].groupby(["cell", "estimator"]).reject.mean().unstack().round(2)
print("\n=== (b) Placebo rejection rates of tau_D (nominal 0.05; check at least within and dml)")
print(pr.to_string())
chk = pr[["within", "dml"]]
ok_b = ((chk >= PLACEBO_RANGE[0]) & (chk <= PLACEBO_RANGE[1])).all().all()
print("-> pass" if ok_b else "-> some cells outside %s (few replications make this noisy)" % (PLACEBO_RANGE,))

# -- (c) convergence
dg = (pol.drop_duplicates(["cell", "rep"]).groupby("cell")[["conv", "profit_vs_nash", "profit_sd"]].mean().round(2))
print("\n=== (c) Learning diagnostics per cell (conv = share of markets with unchanged greedy policy)")
print(dg.to_string())
low = dg[dg.conv < CONV_MIN].index.tolist()
print("-> cells flagged for separate analysis (conv < %.2f): %d of %d" % (CONV_MIN, len(low), len(dg)))
print("\n(d) run time: see the timestamps printed by run_pilot.py")
