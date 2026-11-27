"""Pilot / main experiment runner. Resumable: rows already in the output file are skipped.

Usage:
    python scripts/run_pilot.py --quick                    # tiny smoke test (a few minutes)
    python scripts/run_pilot.py --reps 10 --markets 300    # pilot grid
    N_JOBS=8 python scripts/run_pilot.py --reps 10 --markets 300 --out results/pilot_raw.csv

One row per (cell, replication, estimator, quantity). Truths are computed from paired worlds
in each replication and stored next to the estimates.
"""
import os, sys, csv, time, zlib, argparse, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from cournot_sim import simulate_worlds, truths
from estimators import make_data, run_all

LEARNER = dict(gamma=0.0, floor=0.03, T1=20000)      # calibrated setting (scripts/calibrate_grid.py)
SPEEDS = {"slow": 0.05, "fast": 0.10}
FIELDS = ["cell", "N", "design", "kappa", "speed", "placebo", "rep", "M", "estimator", "quantity",
          "est", "se", "true_DE", "true_TE", "true_SP", "true_GTE", "conv", "profit_vs_nash", "profit_sd"]


def cells(quick, Ns=None, speeds=None, placebos=None):
    Ns = Ns or ([4] if quick else [4, 8])
    designs = ["fixed", "varies"]
    kappas = [0.0, 1.5]
    speeds = speeds or (["slow"] if quick else ["slow", "fast"])
    placebos = [False, True] if placebos is None else placebos
    for N, d, k, s, p in itertools.product(Ns, designs, kappas, speeds, placebos):
        yield dict(N=N, design=d, kappa=k, speed=s, placebo=p)


def cell_id(c):
    return f"N{c['N']}_{c['design']}_k{c['kappa']}_{c['speed']}_{'placebo' if c['placebo'] else 'policy'}"


def n_treat_for(c, M, rng):
    N = c['N']
    if c['design'] == 'fixed':
        return N // 2
    return np.array([max(1, round(p * N)) for p in rng.choice([0.25, 0.5, 0.75], size=M)])


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r['cell'], int(r['rep'])) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--markets", type=int, default=300)
    ap.add_argument("--out", default="results/pilot_raw.csv")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--Ns", default="", help="comma list, e.g. 16 (default: pilot grid)")
    ap.add_argument("--speeds", default="", help="comma list, e.g. slow")
    ap.add_argument("--no-placebo", action="store_true", help="skip placebo cells")
    a = ap.parse_args()
    Ns = [int(x) for x in a.Ns.split(",")] if a.Ns else None
    speeds = a.speeds.split(",") if a.speeds else None
    placebos = [False] if a.no_placebo else None
    if a.quick:
        a.reps, a.markets, a.out = 2, 80, "results/quick_raw.csv"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = done_keys(a.out)
    new = not os.path.exists(a.out)
    t_start = time.time()
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for c in cells(a.quick, Ns, speeds, placebos):
            cid = cell_id(c)
            for rep in range(a.reps):
                if (cid, rep) in done:
                    continue
                seed = (zlib.crc32(cid.encode()) + 1000003 * rep) % (2 ** 31 - 1)
                rng = np.random.default_rng(seed)
                nt = n_treat_for(c, a.markets, rng)
                sim = simulate_worlds(a.markets, c['N'], c['kappa'], nt, SPEEDS[c['speed']], seed=seed,
                                      hetero=True, placebo=c['placebo'], **LEARNER)
                tr = truths(sim)
                nash = (80.0 ** 2) / (c['N'] + 1) ** 2
                diag = dict(conv=sim['conv0'].mean(), profit_vs_nash=sim['Y0'].mean() / nash,
                            profit_sd=sim['Y0'].mean(axis=1).std() / nash)
                res = run_all(make_data(sim), seed=seed)
                for est, qs in res.items():
                    if 'error' in qs:
                        qs = {'ERROR': (np.nan, np.nan)}
                    for q, (e, s) in qs.items():
                        w.writerow(dict(cell=cid, N=c['N'], design=c['design'], kappa=c['kappa'], speed=c['speed'],
                                        placebo=int(c['placebo']), rep=rep, M=a.markets, estimator=est, quantity=q,
                                        est=e, se=s, true_DE=tr['DE'][0], true_TE=tr['TE'][0], true_SP=tr['SP'][0],
                                        true_GTE=tr['GTE'][0], **diag))
                f.flush()
                print(f"[{time.time()-t_start:6.0f}s] {cid} rep {rep+1}/{a.reps}", flush=True)


if __name__ == "__main__":
    main()
