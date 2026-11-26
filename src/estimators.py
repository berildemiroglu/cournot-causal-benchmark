"""Estimators for the benchmark. All take a long-format firm-level data set and return
point estimates with market-clustered standard errors.

Data dict `data` (arrays of length n = M*N, firms stacked market by market):
    y    post-window outcome            pre   burn-in-window outcome (pre-treatment)
    d    treatment indicator            cost  marginal cost (pre-treatment)
    mkt  market id (int)                n_firms  firms per market (for share G)
Derived inside: g = share of treated rivals, market means of pre-treatment covariates.

Every estimator returns a dict  name -> (estimate, se)  for the quantities it can estimate:
    'tau_D'  coefficient on own treatment (what a practitioner would report)
    'gamma_G' coefficient on rivals' treated share (exposure-aware estimators only)
    'GTE'    tau_D + gamma_G, the global-effect estimate (exposure-aware only; NaN if the design
             has no variation in the treated share across markets)
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

NAN = (np.nan, np.nan)
N_JOBS = int(os.environ.get('N_JOBS', 2))   # threads for forests; set N_JOBS=8 on a bigger machine


# ---------------------------------------------------------------- helpers
def make_data(sim):
    """Convert simulate_worlds() output to the long format used here."""
    M, N = sim['D'].shape
    mkt = np.repeat(np.arange(M), N)
    d = sim['D'].ravel()
    return dict(y=sim['Yobs'].ravel(), pre=sim['pre'].ravel(), cost=sim['costs'].ravel(),
                d=d, mkt=mkt, n_firms=np.full(M * N, N))


def rival_share(d, mkt, n_firms):
    tot = np.bincount(mkt, weights=d)[mkt]
    return (tot - d) / (n_firms - 1)


def share_varies(d, mkt):
    """True if the number of treated firms differs across markets (needed to identify gamma_G)."""
    tot = np.bincount(mkt, weights=d)
    return np.ptp(tot) > 0


def features(data):
    """Pre-treatment covariates for nuisance models: own cost, own pre-profit, market means."""
    mkt = data['mkt']
    cnt = np.bincount(mkt)
    mc = (np.bincount(mkt, weights=data['cost']) / cnt)[mkt]
    mp = (np.bincount(mkt, weights=data['pre']) / cnt)[mkt]
    return np.column_stack([data['cost'], data['pre'], mc, mp])


def demean(x, mkt):
    x = np.asarray(x, float)
    cnt = np.bincount(mkt)
    if x.ndim == 1:
        return x - (np.bincount(mkt, weights=x) / cnt)[mkt]
    return x - np.column_stack([np.bincount(mkt, weights=x[:, j]) / cnt for j in range(x.shape[1])])[mkt]


def cluster_ols(X, y, mkt, dof_k=None):
    """OLS with cluster-robust (CR1) covariance. Returns beta, V."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    scores = X * u[:, None]
    G = mkt.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, mkt, scores)
    meat = S.T @ S
    kk = dof_k if dof_k is not None else k
    c = G / (G - 1) * (n - 1) / max(n - kk, 1)
    return beta, c * XtX_inv @ meat @ XtX_inv


def _pick(beta, V, j):
    return float(beta[j]), float(np.sqrt(V[j, j]))


def _combo(beta, V, w):
    w = np.asarray(w, float)
    return float(w @ beta), float(np.sqrt(w @ V @ w))


# ---------------------------------------------------------------- 1-3: regression family
def naive(data):
    X = np.column_stack([np.ones_like(data['d']), data['d']])
    b, V = cluster_ols(X, data['y'], data['mkt'])
    return {'tau_D': _pick(b, V, 1)}


def within(data):
    """Market fixed effects + own cost and own pre-treatment profit."""
    mkt = data['mkt']
    X = demean(np.column_stack([data['d'], data['cost'], data['pre']]), mkt)
    y = demean(data['y'], mkt)
    G = mkt.max() + 1
    b, V = cluster_ols(X, y, mkt, dof_k=X.shape[1] + G)
    return {'tau_D': _pick(b, V, 0)}


def did(data):
    """Difference-in-differences: (post - pre window) on D with market fixed effects."""
    mkt = data['mkt']
    X = demean(data['d'][:, None], mkt)
    y = demean(data['y'] - data['pre'], mkt)
    G = mkt.max() + 1
    b, V = cluster_ols(X, y, mkt, dof_k=1 + G)
    return {'tau_D': _pick(b, V, 0)}


def ols_exposure(data):
    """Linear exposure-aware benchmark: y on d, g, own covariates, no market fixed effects."""
    d, mkt = data['d'], data['mkt']
    g = rival_share(d, mkt, data['n_firms'])
    X = np.column_stack([np.ones_like(d), d, g, data['cost'], data['pre']])
    b, V = cluster_ols(X, data['y'], mkt)
    out = {'tau_D': _pick(b, V, 1)}
    if share_varies(d, mkt):
        out['gamma_G'] = _pick(b, V, 2)
        out['GTE'] = _combo(b, V, [0, 1, 1, 0, 0])
    else:
        out['gamma_G'] = NAN
        out['GTE'] = NAN
    return out


# ---------------------------------------------------------------- 4-6: orthogonal learners
def _rf(seed):
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.7,
                                 n_jobs=N_JOBS, random_state=seed)


def crossfit_residuals(data, treat_cols, n_splits=5, seed=0):
    """Out-of-fold residuals of y and of each treatment column on X, folds = whole markets."""
    X = features(data)
    mkt = data['mkt']
    T = np.column_stack(treat_cols)
    ytil = np.empty(len(mkt))
    Ttil = np.empty_like(T, dtype=float)
    for tr, te in GroupKFold(n_splits).split(X, groups=mkt):
        ytil[te] = data['y'][te] - _rf(seed).fit(X[tr], data['y'][tr]).predict(X[te])
        for j in range(T.shape[1]):
            Ttil[te, j] = T[te, j] - _rf(seed + 1 + j).fit(X[tr], T[tr, j]).predict(X[te])
    return ytil, Ttil, X


def dml(data, exposure=False, seed=0):
    """Partially linear double ML with market-clustered cross-fitting and clustered sandwich SE."""
    d, mkt = data['d'], data['mkt']
    g = rival_share(d, mkt, data['n_firms'])
    use_g = exposure and share_varies(d, mkt)
    cols = [d, g] if use_g else [d]
    ytil, Ttil, _ = crossfit_residuals(data, cols, seed=seed)
    b, V = cluster_ols(Ttil, ytil, mkt)
    out = {'tau_D': _pick(b, V, 0)}
    if exposure:
        if use_g:
            out['gamma_G'] = _pick(b, V, 1)
            out['GTE'] = _combo(b, V, [1, 1])
        else:
            out['gamma_G'] = NAN
            out['GTE'] = NAN
    return out


def causal_forest(data, exposure=False, seed=0, n_estimators=400):
    """Causal forest (econml GRF) on market-cross-fitted residuals. The ATE is the debiased average
    of out-of-bag effect predictions; its SE is clustered by market. Note: the forest's own
    subsampling and honesty splits are by firm, not by market (documented limitation)."""
    from econml.grf import CausalForest
    d, mkt = data['d'], data['mkt']
    g = rival_share(d, mkt, data['n_firms'])
    use_g = exposure and share_varies(d, mkt)
    cols = [d, g] if use_g else [d]
    ytil, Ttil, X = crossfit_residuals(data, cols, seed=seed)
    cf = CausalForest(n_estimators=n_estimators, min_samples_leaf=10, max_samples=0.45,
                      honest=True, random_state=seed, n_jobs=N_JOBS)
    cf.fit(X, Ttil, ytil)
    theta = cf.oob_predict(X)
    theta = theta.reshape(len(ytil), -1)
    k = Ttil.shape[1]
    A = np.linalg.inv(Ttil.T @ Ttil / len(ytil))
    resid = ytil - np.sum(theta * Ttil, axis=1)
    psi = theta + (Ttil * resid[:, None]) @ A.T            # debiased score for each treatment effect
    G = mkt.max() + 1

    def stat(w):
        s = psi @ np.asarray(w, float)
        est = s.mean()
        cl = np.zeros(G)
        np.add.at(cl, mkt, s - est)
        return float(est), float(np.sqrt((cl ** 2).sum() * G / (G - 1)) / len(s))

    out = {'tau_D': stat([1] + [0] * (k - 1))}
    if exposure:
        if use_g:
            out['gamma_G'] = stat([0, 1])
            out['GTE'] = stat([1, 1])
        else:
            out['gamma_G'] = NAN
            out['GTE'] = NAN
    return out


ESTIMATORS = {
    'naive': naive,
    'within': within,
    'did': did,
    'ols_exposure': ols_exposure,
    'dml': lambda data, seed=0: dml(data, exposure=False, seed=seed),
    'cforest': lambda data, seed=0: causal_forest(data, exposure=False, seed=seed),
    'dml_exposure': lambda data, seed=0: dml(data, exposure=True, seed=seed),
    'cforest_exposure': lambda data, seed=0: causal_forest(data, exposure=True, seed=seed),
}


def run_all(data, seed=0):
    rows = {}
    for name, f in ESTIMATORS.items():
        try:
            rows[name] = f(data, seed=seed) if name.startswith(('dml', 'cforest')) else f(data)
        except Exception as e:                     # keep the run going, record the failure
            rows[name] = {'error': str(e)}
    return rows
