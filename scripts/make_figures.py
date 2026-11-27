"""Figures for the paper from the main run.  Usage: python scripts/make_figures.py results/main_raw.csv figures/
Palette: reference categorical slots 1-4 (blue, orange, aqua, yellow) on a white surface; identity is also carried
by marker shape and by direct labels, never by color alone."""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/main_raw.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(out, exist_ok=True)
df = pd.read_csv(src)
df = df[df.est.notna() & (df.placebo == 0) & (df.quantity != "ERROR")].copy()
df["true_TESP"] = df.true_TE - df.true_SP
Z = 1.96
C = {"blue": "#257de0", "orange": "#f47230", "aqua": "#00b894", "yellow": "#f2a600"}
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white"})
NS = [4, 6, 8]

def cell_stats(est, quantity, target):
    rows = []
    for cell, d in df[(df.estimator == est) & (df.quantity == quantity)].groupby("cell"):
        tv = df[(df.cell == cell)].drop_duplicates("rep")[target].mean()
        e = d.est - tv
        rows.append(dict(N=d.N.iloc[0], design=d.design.iloc[0], kappa=d.kappa.iloc[0],
                         rb=e.mean() / tv, ci=Z * e.std() / np.sqrt(len(d)) / abs(tv)))
    return pd.DataFrame(rows)

# Figure 1: size of the gap DE vs TE-SP by N
t = df.drop_duplicates(["cell", "rep"]).copy()
t["gap"] = ((t.true_DE - t.true_TESP) / t.true_DE).abs()
g = t.groupby("N").gap.agg(["mean", "std", "count"])
fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.errorbar(g.index, g["mean"] * 100, yerr=Z * g["std"] / np.sqrt(g["count"]) * 100, color=C["blue"], marker="o", ms=8,
            lw=2, capsize=3, mec="white", mew=1.5)
ax.set_xticks(NS); ax.set_xlabel("Number of firms N"); ax.set_ylabel("|DE - (TE - SP)| / DE  (%)")
ax.set_ylim(0, 30); fig.tight_layout(); fig.savefig(f"{out}/fig1_gap_by_N.pdf"); fig.savefig(f"{out}/fig1_gap_by_N.png", dpi=200); plt.close(fig)

# Figure 2: relative bias of tau_D against DE, by estimator, panels = design x kappa
ests = [("naive", "Naive", C["yellow"], "s"), ("within", "Within", C["orange"], "^"),
        ("did", "DiD", C["aqua"], "D"), ("dml", "DML / forest", C["blue"], "o")]
fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True, sharey=True)
for i, design in enumerate(["fixed", "varies"]):
    for j, kappa in enumerate([0.0, 1.5]):
        ax = axs[i, j]
        ax.axhspan(-10, 10, color="#f1f1f1", zorder=0)
        for est, lab, col, mk in ests:
            s = cell_stats(est, "tau_D", "true_DE"); s = s[(s.design == design) & (s.kappa == kappa)].sort_values("N")
            ax.errorbar(s.N, s.rb * 100, yerr=s.ci * 100, color=col, marker=mk, ms=7, lw=1.8, capsize=2, mec="white", mew=1, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{'Fixed' if design == 'fixed' else 'Varying'} treated share, kappa = {kappa}", fontsize=9, loc="left")
        ax.set_xticks(NS)
        if i == 1: ax.set_xlabel("Number of firms N")
        if j == 0: ax.set_ylabel("Relative bias vs DE (%)")
axs[0, 0].legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
fig.tight_layout(rect=(0, 0, 1, 1)); fig.savefig(f"{out}/fig2_bias_vs_DE.pdf"); fig.savefig(f"{out}/fig2_bias_vs_DE.png", dpi=200); plt.close(fig)

# Figure 3: GTE recovery with exposure-aware estimators (share-varying designs)
ge = [("dml_exposure", "DML, exposure-aware", C["blue"], "o"), ("cforest_exposure", "Forest, exposure-aware", C["aqua"], "D"),
      ("ols_exposure", "OLS, exposure", C["orange"], "^")]
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
for j, kappa in enumerate([0.0, 1.5]):
    ax = axs[j]; ax.axhspan(-10, 10, color="#f1f1f1", zorder=0)
    for k, (est, lab, col, mk) in enumerate(ge):
        s = cell_stats(est, "GTE", "true_GTE"); s = s[(s.design == "varies") & (s.kappa == kappa)].sort_values("N")
        ax.errorbar(s.N + (k - 1) * 0.12, s.rb * 100, yerr=s.ci * 100, color=col, marker=mk, ms=7, lw=1.8, capsize=2, mec="white", mew=1, label=lab)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(NS); ax.set_xlabel("Number of firms N")
    ax.set_title(f"Varying share, kappa = {kappa}", fontsize=9, loc="left")
    if j == 0: ax.set_ylabel("Relative bias vs GTE (%)")
axs[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(f"{out}/fig3_GTE_exposure.pdf"); fig.savefig(f"{out}/fig3_GTE_exposure.png", dpi=200); plt.close(fig)
print("saved to", out)
