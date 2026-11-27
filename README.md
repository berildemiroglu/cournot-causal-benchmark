# Which policy effects do causal estimators recover? An agent-based Cournot market with learning firms

Code, protocol and analysis scripts for the paper *Which Policy Effects Do Causal Estimators Recover? Evidence from an Agent-Based Cournot Market with Learning Firms* (`paper/paper.tex`).

The simulator is a repeated Cournot market with Q-learning firms. A per-unit subsidy is given to some firms. For every market, the simulator also runs paired counterfactual worlds with common random numbers, so the true effects are known:

| Effect | Meaning |
|---|---|
| DE | direct effect of own treatment, rivals held fixed |
| TE | total effect on treated firms |
| SP | spillover on untreated firms |
| GTE | global effect: everyone treated vs. nobody treated |

Seven estimators are compared against these truths: naive contrast, within-market regression, difference-in-differences, double machine learning (DML), causal forest, exposure-aware DML and forest, and OLS with exposure.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The paper's causal forest results use `econml==0.17.1`. If `econml` is missing, `run_pilot.py` does not stop: the forest rows are written as `ERROR` in the output file. Check that `econml` is installed before a long run.

## Quick check

```bash
python tests/test_sim.py                  # Nash benchmark, placebo = 0, reproducibility
python scripts/run_pilot.py --quick       # small smoke test of the whole pipeline (a few minutes)
```

## Reproducing the paper

The runner is resumable: rows already in the output file are skipped, so an interrupted run can be restarted with the same command. `N_JOBS=8` sets the number of parallel jobs.

**1. Learner calibration (Appendix A).**

```bash
python scripts/calibrate_grid.py          # one-seed grid at N=6 (calibration grid table of Appendix A)
python scripts/check_calibrated.py        # chosen setting at N=4, 8, 16 (check table of Appendix A)
```

**2. Main run (confirmatory set: slow learners, N = 4, 6, 8).** 24 cells, 50 repetitions of 300 markets.

```bash
N_JOBS=8 python scripts/run_pilot.py --reps 50 --markets 300 --Ns 4,6,8 --speeds slow \
    --seed-offset 5000000 --out results/main_raw.csv
python scripts/analyze_main.py results/main_raw.csv > results/main_hypotheses.txt
python scripts/make_tables.py results/main_raw.csv paper/tables.tex
python scripts/make_figures.py results/main_raw.csv paper/figures
```

**3. Exploratory runs (Section 5.7).** These are not part of the confirmatory set: fast learners and slow learners at N = 12 and 16 did not reach the convergence criterion of the protocol.

```bash
# slow learners, larger markets (20 repetitions)
N_JOBS=8 python scripts/run_pilot.py --reps 20 --markets 300 --Ns 12,16 --speeds slow \
    --seed-offset 7000000 --out results/explore_big_raw.csv
# fast learners, N = 4, 6, 8 (20 repetitions)
N_JOBS=8 python scripts/run_pilot.py --reps 20 --markets 300 --Ns 4,6,8 --speeds fast \
    --out results/explore_fast_raw.csv
```
<!-- TODO before publishing: check the exact flags of the exploratory runs in the shell history (especially --seed-offset for the fast-learner run) and the main run. -->

Each row of a raw results file is one (cell, replication, estimator, quantity). The true effects of the replication are stored next to the estimates.

## Protocol

`PROTOCOL.md` fixes the design, estimators, decision thresholds, smallest effect of interest (10% of the true effect) and hypotheses H1 to H5. It was committed (`cc25fa0`, 27 November 2026) before the pilot run. Addendum 1 (after the pilot, before the main run) restricts the confirmatory set to slow learners at N = 4, 6, 8, because the other cells did not meet the convergence criterion. The commit history is a timestamp kept by the authors, not an independent registry.

## Layout

- `src/cournot_sim.py`: simulator (numba, vectorized across markets, common random numbers)
- `src/estimators.py`: the estimators (naive, within, DiD, DML, causal forest, exposure-aware versions, OLS exposure)
- `scripts/run_pilot.py`: experiment runner (pilot, main run and exploratory runs)
- `scripts/analyze_pilot.py`, `scripts/analyze_main.py`: pilot checks and the hypothesis tests of the main run
- `scripts/calibrate_grid.py`, `scripts/check_calibrated.py`, `scripts/calibrate_learners.py`: calibration diagnostics
- `scripts/make_tables.py`, `scripts/make_figures.py`: tables and figures of the paper
- `scripts/demo_estimators.py`: all estimators on one simulated data set
- `tests/`: checks of the simulator
- `results/`: raw and summary results
- `paper/`: LaTeX source of the paper

## Limits of the results

The results are for one stylized market and one calibration (linear demand, stateless learners with discount 0, one policy). Fast learners and markets above N = 8 did not meet the convergence criterion, so they are reported as exploratory. The paper discusses these limits in Section 7.

## Citation

If you use this code, please cite the paper (details to be added after publication).
