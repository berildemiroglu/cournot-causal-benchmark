"""LaTeX tables for the paper from the main run. Usage: python scripts/make_tables.py results/main_raw.csv paper/tables.tex"""
import sys
import numpy as np, pandas as pd
src = sys.argv[1]; out = sys.argv[2]
df = pd.read_csv(src); df = df[df.est.notna() & (df.placebo == 0) & (df.quantity != "ERROR")].copy()
df["true_TESP"] = df.true_TE - df.true_SP
Z = 1.96
reps = df.drop_duplicates(["cell", "rep"])
tr = reps.groupby(["N", "design", "kappa"])[["true_DE", "true_TE", "true_SP", "true_GTE", "true_TESP", "conv", "profit_vs_nash"]].mean()
L = []
# Table A: truths by N (design fixed, kappa 0 as reference)
L.append(r"\begin{table}[t]\centering\small\caption{True effects and learning diagnostics (slow learners, fixed share, $\kappa=0$; mean over 50 repetitions of 300 markets).}\label{tab:truth}")
L.append(r"\begin{tabular}{rrrrrrrr}\toprule $N$ & DE & TE & SP & GTE & TE$-$SP & Converged & Profit/Nash\\\midrule")
for N in [4, 6, 8]:
    r = tr.loc[(N, "fixed", 0.0)]
    L.append(f"{N} & {r.true_DE:.1f} & {r.true_TE:.1f} & {r.true_SP:.1f} & {r.true_GTE:.1f} & {r.true_TESP:.1f} & {r.conv:.2f} & {r.profit_vs_nash:.2f}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
# Table B: relative bias of tau_D vs DE and TE-SP
def rb(est, q, target, N, design, kappa):
    d = df[(df.estimator == est) & (df.quantity == q) & (df.N == N) & (df.design == design) & (df.kappa == kappa)]
    tv = d.drop_duplicates("rep")[target].mean() if target in d else np.nan
    tv = reps[(reps.N == N) & (reps.design == design) & (reps.kappa == kappa)][target].mean()
    return (d.est - tv).mean() / tv
L.append(r"\begin{table}[t]\centering\small\caption{Relative bias (in percent) of the own-treatment coefficient against DE and against TE$-$SP. Slow learners, 50 repetitions per cell. Entries within 10\% in absolute value are the smallest-effect-of-interest cases.}\label{tab:bias}")
L.append(r"\begin{tabular}{llrrrrrr}\toprule & & \multicolumn{3}{c}{vs DE} & \multicolumn{3}{c}{vs TE$-$SP}\\\cmidrule(lr){3-5}\cmidrule(lr){6-8} Design, $\kappa$ & Estimator & $N=4$ & $N=6$ & $N=8$ & $N=4$ & $N=6$ & $N=8$\\\midrule")
names = {"naive": "Naive", "within": "Within", "did": "DiD", "dml": "DML", "cforest": "Forest"}
for design in ["fixed", "varies"]:
    for kappa in [0.0, 1.5]:
        for i, e in enumerate(["naive", "within", "did", "dml", "cforest"]):
            lab = f"{'Fixed' if design=='fixed' else 'Varying'}, {kappa}" if i == 0 else ""
            vals = [100 * rb(e, "tau_D", "true_DE", N, design, kappa) for N in [4, 6, 8]] + [100 * rb(e, "tau_D", "true_TESP", N, design, kappa) for N in [4, 6, 8]]
            L.append(f"{lab} & {names[e]} & " + " & ".join(f"{v:.0f}" for v in vals) + r"\\")
        L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}\end{table}"
# Table C: GTE with exposure-aware estimators
L.append(r"\begin{table}[t]\centering\small\caption{Global effect from exposure-aware estimators (designs with varying treated share, slow learners). Relative bias in percent and coverage of the nominal 95\% interval.}\label{tab:gte}")
L.append(r"\begin{tabular}{llrrrrrr}\toprule & & \multicolumn{3}{c}{Relative bias (\%)} & \multicolumn{3}{c}{Coverage}\\\cmidrule(lr){3-5}\cmidrule(lr){6-8} $\kappa$ & Estimator & $N=4$ & $N=6$ & $N=8$ & $N=4$ & $N=6$ & $N=8$\\\midrule")
gn = {"dml_exposure": "DML, exposure-aware", "cforest_exposure": "Forest, exposure-aware", "ols_exposure": "OLS, exposure"}
for kappa in [0.0, 1.5]:
    for i, e in enumerate(["dml_exposure", "cforest_exposure", "ols_exposure"]):
        b, c = [], []
        for N in [4, 6, 8]:
            d = df[(df.estimator == e) & (df.quantity == "GTE") & (df.N == N) & (df.design == "varies") & (df.kappa == kappa)]
            tv = reps[(reps.N == N) & (reps.design == "varies") & (reps.kappa == kappa)].true_GTE.mean()
            b.append(100 * (d.est - tv).mean() / tv)
            c.append(((d.est - Z * d.se <= tv) & (tv <= d.est + Z * d.se)).mean())
        L.append(f"{kappa if i == 0 else ''} & {gn[e]} & " + " & ".join(f"{v:.0f}" for v in b) + " & " + " & ".join(f"{v:.2f}" for v in c) + r"\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}\end{table}"
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
