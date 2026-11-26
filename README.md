# Causal estimators in markets with feedback

Agent-based Cournot market with Q-learning firms and known counterfactuals.


## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python tests/test_sim.py              # Nash benchmark, placebo = 0, reproducibility
python scripts/calibrate_learners.py  # learner calibration diagnostic (Section 5.1)
```

## Layout
- `src/cournot_sim.py` simulator (numba, vectorized across markets, common random numbers)
- `tests/` checks, `scripts/` diagnostics

## Status
Week 1 of the one-month plan: simulator done; estimators, calibration and literature check next.
