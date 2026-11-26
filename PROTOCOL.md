# Protocol (frozen 27 November 2026, before the pilot run)

Study: which policy effect do causal estimators recover in an agent-based Cournot market with Q-learning firms and known counterfactuals.

## Design
- Market: inverse demand P = 100 - Q, marginal cost 20 + U[-3, 3], per-unit subsidy s = 5, quantity grid K = 21.
- Learners: stateless Q-learning, gamma = 0, eps_min = 0.03, burn-in T0 = 2500, total T1 = 20000, windows W = 500. Slow alpha = 0.05, fast alpha = 0.10.
- Counterfactual worlds with common random numbers: no policy, realized assignment, all treated, flip world.
- Estimands: DE (direct), TE (total on treated), SP (spillover on untreated), GTE (all treated vs none).

## Pilot grid (32 cells)
N in {4, 8} x design in {fixed share, varying share} x kappa in {0, 1.5} x speed in {slow, fast} x placebo in {no, yes}; 10 repetitions, 300 markets per repetition.
Main run: full factorial, adds N = 16. Pilot data are not pooled with the main run.

## Estimators
naive difference, within, DiD, ols_exposure, DML, causal forest, exposure-aware DML and forest. Market-clustered standard errors and cross-fitting.

## Metrics
Bias, RMSE and 95% coverage against each estimand; placebo rejection rate; convergence share.

## Decision thresholds (fixed here)
- SESOI: 10% of the true value of the estimand in that cell.
- Go/no-go after the pilot:
  (a) Monte Carlo s.e. of the truth / |truth| < 0.10;
  (b) placebo rejection rate between 0.02 and 0.10 for within and DML;
  (c) share of markets with converged greedy policies >= 0.50 in every cell;
  (d) run time of the main run feasible within the schedule.
- If a check fails, the main run uses a reduced grid. No second pilot.

## Hypotheses
- H1. Adjusted within-market estimators recover DE with bias below the SESOI in most cells, but their relative gap to GTE exceeds the SESOI in most cells.
- H2. Estimators recover TE and GTE only when the treated share varies across markets and the estimator uses it. With a fixed share they cannot, whatever the learner. (The design idea itself is known: Baird et al. 2018; Munro, Kuang and Wager 2025.)
- H3. Coverage of market-clustered intervals falls below 90% in some cells with fast learning or small markets.
- H4. Differences between DML and causal forest are smaller than differences between designs that do and do not vary the treated share.
- H5. The gap between DE and TE - SP shrinks as N grows.

Tests: regression of bias on the factors, Benjamini-Hochberg correction across H1-H5.

## Known limitations (stated in advance)
Stateless learners (gamma = 0); causal forest honesty is by firm, not by market; thresholds above were set from the calibration runs and one demo seed.
