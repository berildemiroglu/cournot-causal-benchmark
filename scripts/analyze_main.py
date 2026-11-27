"""Main-run analysis: hypotheses H1-H5 (PROTOCOL.md + Addendum 1), SESOI = 10% of the true effect.

Usage: python scripts/analyze_main.py results/main_raw.csv > results/main_hypotheses.txt
Operationalisations (fixed here, written after seeing the report but before looking at test results):
  H1  estimators within, did, dml, cforest (tau_D): |rel. bias vs DE| < 0.10 in most (>50%) of estimator x cell pairs,
      AND |rel. bias vs GTE| > 0.10 in most pairs. p = max of the two exact one-sided binomial p-values (cells treated
      as independent units: approximate).
  H2  exposure-aware GTE (dml_exposure, cforest_exposure) has |rel. bias| < 0.10 in most share-varying pairs, while no
      estimator is within 0.10 of GTE in fixed-share cells (GTE not estimable there). TE is not estimated by the
      exposure-aware estimators in this run, so the TE part of H2 is not tested. p = exact binomial for the varying part.
  H3  coverage (95% nominal) below 0.90 in some cell: exact one-sided binomial test per estimator x cell against 0.90
      (own target: tau_D vs DE for dml/cforest; GTE vs GTE for exposure-aware GTE); p = Bonferroni-adjusted minimum.
  H4  mean |dml - cforest| (tau_D) smaller than the mean |difference between fixed and varying designs|, matched on N, kappa;
      bootstrap over replications within cell.
  H5  absolute gap |DE - (TE-SP)| / DE decreases with N: OLS on replication-level truths with factors N, kappa, design; one-sided.
All five p-values go through Benjamini-Hochberg.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else "results/main_raw.csv"
SESOI = 0.10
Z = 1.96
rng = np.random.default_rng(20261127)
df = pd.read_csv(path)
df = df[df.est.notna() & (df.quantity != "ERROR")]
pol = df[df.placebo == 0].copy()
pol["true_TESP"] = pol.true_TE - pol.true_SP
cellt = pol.drop_duplicates(["cell", "rep"]).groupby("cell")[["true_DE", "true_TE", "true_SP", "true_GTE", "true_TESP"]].mean()
cellinfo = pol.drop_duplicates("cell").set_index("cell")[["N", "design", "kappa"]]

def stat(estimator, quantity, target):
    out = []
    for cell, d in pol[(pol.estimator == estimator) & (pol.quantity == quantity)].groupby("cell"):
        tv = cellt.loc[cell, target]
        e = d.est - tv
        cov = ((d.est - Z * d.se <= tv) & (tv <= d.est + Z * d.se)).mean()
        out.append(dict(cell=cell, estimator=estimator, target=target, n=len(d), truth=tv,
                        rel_bias=e.mean() / tv, rel_mcse=e.std() / np.sqrt(len(d)) / abs(tv), coverage=cov,
                        k_cover=int(round(cov * len(d)))))
    return pd.DataFrame(out).join(cellinfo, on="cell")

def binom_ge(k, n, p=0.5):
    return stats.binomtest(int(k), int(n), p, alternative="greater").pvalue

print("=== H1: adjusted estimators (within, did, dml, cforest), tau_D")
rows = []
for e in ["within", "did", "dml", "cforest"]:
    de = stat(e, "tau_D", "true_DE"); gt = stat(e, "tau_D", "true_GTE"); ts = stat(e, "tau_D", "true_TESP")
    rows.append(pd.DataFrame(dict(estimator=e, cell=de.cell, relbias_DE=de.rel_bias, relbias_TESP=ts.rel_bias, relbias_GTE=gt.rel_bias)))
h1 = pd.concat(rows)
h1["pass_DE"] = h1.relbias_DE.abs() < SESOI
h1["gap_GTE"] = h1.relbias_GTE.abs() > SESOI
print(h1.groupby("estimator")[["pass_DE", "gap_GTE"]].agg(["sum", "count"]).to_string())
print("share of cells within 10%% of TE-SP: %s" % (h1.assign(p=h1.relbias_TESP.abs() < SESOI).groupby("estimator").p.sum().to_dict()))
nA, kA = len(h1), h1.pass_DE.sum()
nB, kB = len(h1), h1.gap_GTE.sum()
pA, pB = binom_ge(kA, nA), binom_ge(kB, nB)
p1 = max(pA, pB)
print(f"part A (DE recovered within 10%): {kA}/{nA}, p={pA:.4f};  part B (GTE gap > 10%): {kB}/{nB}, p={pB:.4g};  H1 p={p1:.4g}")
print(h1[h1.estimator.isin(["within", "dml"])].round(2).to_string(index=False))

print("\n=== H2: exposure-aware GTE")
g2 = pd.concat([stat(e, "GTE", "true_GTE") for e in ["dml_exposure", "cforest_exposure"]])
g2 = g2[g2.design == "varies"]
g2["pass"] = g2.rel_bias.abs() < SESOI
print(g2[["cell", "estimator", "truth", "rel_bias", "rel_mcse", "coverage", "pass"]].round(3).to_string(index=False))
nf = pol[(pol.design == "fixed") & (pol.estimator == "dml_exposure") & (pol.quantity == "GTE")].est.notna().sum()
fixed_tau = pd.concat([stat(e, "tau_D", "true_GTE") for e in ["within", "did", "dml", "cforest", "naive"]])
fixed_tau = fixed_tau[fixed_tau.design == "fixed"]
print(f"fixed-share cells: exposure-aware GTE estimates available: {nf}; tau_D within 10% of GTE in {(fixed_tau.rel_bias.abs() < SESOI).sum()} of {len(fixed_tau)} estimator-cell pairs")
k2, n2 = g2["pass"].sum(), len(g2)
p2 = binom_ge(k2, n2)
print(f"varying-share GTE within 10%: {k2}/{n2}, p={p2:.4f}  (TE part of H2 not tested in this run)")

print("\n=== H3: coverage below 0.90 (own target)")
c3 = pd.concat([stat("dml", "tau_D", "true_DE"), stat("cforest", "tau_D", "true_DE"),
                stat("dml_exposure", "GTE", "true_GTE"), stat("cforest_exposure", "GTE", "true_GTE")])
c3["target"] = c3.target.where(c3.estimator.isin(["dml", "cforest"]), "GTE")
c3["p_below90"] = [stats.binomtest(k, n, 0.90, alternative="less").pvalue for k, n in zip(c3.k_cover, c3.n)]
print(c3[["cell", "estimator", "target", "coverage", "p_below90"]].round(3).sort_values("p_below90").head(14).to_string(index=False))
p3 = min(1.0, c3.p_below90.min() * len(c3))
print(f"cells with coverage < 0.90: {(c3.coverage < 0.90).sum()} of {len(c3)}; Bonferroni-adjusted min p = {p3:.4f}")

print("\n=== H4: learner differences vs design differences (tau_D, no exposure)")
w = pol[(pol.quantity == "tau_D") & pol.estimator.isin(["dml", "cforest"])].pivot_table(index=["cell", "rep"], columns="estimator", values="est")
w = w.reset_index().join(cellinfo, on="cell")
def stats_h4(w):
    ld = (w.dml - w.cforest).abs().mean()
    m = w.groupby(["N", "kappa", "design"]).dml.mean().unstack("design")
    dd = (m["fixed"] - m["varies"]).abs().mean()
    return ld, dd
ld, dd = stats_h4(w)
boots = []
for _ in range(2000):
    ws = w.groupby("cell", group_keys=False).apply(lambda d: d.sample(len(d), replace=True, random_state=int(rng.integers(1 << 30))))
    a, b = stats_h4(ws); boots.append(b - a)
p4 = (np.array(boots) <= 0).mean()
print(f"mean |dml - cforest| = {ld:.3f}; mean |design difference in dml| = {dd:.3f}; bootstrap p(design diff <= learner diff) = {p4:.4f}")

print("\n=== H5: gap between DE and TE - SP vs N (replication-level truths)")
t = pol.drop_duplicates(["cell", "rep"]).copy()
t["gap"] = ((t.true_DE - t.true_TESP) / t.true_DE).abs()   # H5 is about the size of the gap (it is negative here)
t["varies"] = (t.design == "varies").astype(float)
t["kappa"] = t.kappa.astype(float)
import numpy.linalg as la
X = np.column_stack([np.ones(len(t)), t.N, t.kappa, t.varies]); y = t.gap.values
beta = la.lstsq(X, y, rcond=None)[0]; res = y - X @ beta
cl = t.cell.values
meat = sum(np.outer(X[cl == c].T @ res[cl == c], X[cl == c].T @ res[cl == c]) for c in np.unique(cl))
V = la.inv(X.T @ X) @ meat @ la.inv(X.T @ X)
se = np.sqrt(np.diag(V)); tstat = beta / se
p5 = stats.t.cdf(tstat[1], df=len(np.unique(cl)) - 1)
print(f"slope of gap on N: {beta[1]:.4f} (se {se[1]:.4f}), one-sided p (slope < 0) = {p5:.4g}")
print(t.groupby("N").gap.agg(["mean", "std"]).round(3).to_string())

print("\n=== Benjamini-Hochberg across H1-H5")
ps = pd.Series({"H1": p1, "H2": p2, "H3": p3, "H4": p4, "H5": p5})
o = ps.sort_values(); m = len(o)
adj = (o * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
print(pd.DataFrame({"p": o, "BH_adj": adj}).round(4).to_string())
