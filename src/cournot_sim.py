"""Cournot market with Q-learning firms: simulator with paired counterfactual worlds.

Timeline per market: burn-in (no policy) -> assignment at T0 -> post period.
Worlds (all share the same burn-in Q-tables and, via per-market seeds, the same
post-period random draws = common random numbers):
    W0   no policy
    Wobs realized assignment D
    Wall everyone treated
    Wflip realized assignment with one randomly chosen firm's treatment flipped
"""
import numpy as np
from numba import njit, prange

A_DEM = 100.0   # inverse demand intercept
B_DEM = 1.0     # inverse demand slope
K_ACT = 21      # quantity grid size


def qgrid(N, cbar=20.0, s=5.0):
    qmax = 2.0 * (A_DEM - cbar + s) / (B_DEM * (N + 1))
    return np.linspace(0.0, qmax, K_ACT)


@njit(cache=True)
def _step_profits(acts, grid, c, d, s, out):
    N = acts.shape[0]
    Q = 0.0
    for i in range(N):
        Q += grid[acts[i]]
    P = A_DEM - B_DEM * Q
    if P < 0.0:
        P = 0.0
    for i in range(N):
        out[i] = (P - c[i] + s * d[i]) * grid[acts[i]]


@njit(cache=True)
def _run_periods(Q, grid, c, d, s, t0, t1, alpha, gamma, beta, floor, win_start, prof_acc):
    """Run periods t0..t1-1 in place on Q[N,K]; accumulate profits for t>=win_start."""
    N, K = Q.shape
    acts = np.empty(N, np.int64)
    prof = np.empty(N)
    cnt = 0
    for t in range(t0, t1):
        eps = np.exp(-beta * t)
        if eps < floor:
            eps = floor
        for i in range(N):
            u = np.random.random()
            r = np.random.randint(0, K)       # always drawn: keeps the stream aligned (CRN)
            if u < eps:
                acts[i] = r
            else:
                best = 0
                bv = Q[i, 0]
                for k in range(1, K):
                    if Q[i, k] > bv:
                        bv = Q[i, k]
                        best = k
                acts[i] = best
        _step_profits(acts, grid, c, d, s, prof)
        for i in range(N):
            mx = Q[i, 0]
            for k in range(1, K):
                if Q[i, k] > mx:
                    mx = Q[i, k]
            a = acts[i]
            Q[i, a] += alpha * (prof[i] + gamma * mx - Q[i, a])
        if t >= win_start:
            for i in range(N):
                prof_acc[i] += prof[i]
            cnt += 1
    for i in range(N):
        prof_acc[i] /= max(cnt, 1)


@njit(parallel=True, cache=True)
def burnin(costs, grid, seeds, T0, W, alpha, gamma, beta, floor):
    M, N = costs.shape
    Q = np.zeros((M, N, grid.shape[0]))
    pre = np.zeros((M, N))
    zero = np.zeros(N)
    for m in prange(M):
        np.random.seed(seeds[m])
        for i in range(N):
            for k in range(grid.shape[0]):
                Q[m, i, k] = 1e-6 * np.random.random()
        _run_periods(Q[m], grid, costs[m], zero, 0.0, 0, T0, alpha, gamma, beta, floor, T0 - W, pre[m])
    return Q, pre


@njit(parallel=True, cache=True)
def post(Q0, costs, D, grid, seeds_post, s, T0, T1, W, alpha, gamma, beta, floor):
    """Run post period for assignment matrix D[M,N] from burn-in state Q0 (copied)."""
    M, N = costs.shape
    out = np.zeros((M, N))
    conv = np.zeros(M)
    for m in prange(M):
        Q = Q0[m].copy()
        np.random.seed(seeds_post[m])
        # greedy policy at start of outcome window vs end -> convergence diagnostic
        _run_periods(Q, grid, costs[m], D[m], s, T0, T1 - W, alpha, gamma, beta, floor, T1 + 1, out[m])
        g0 = np.empty(N, np.int64)
        for i in range(N):
            g0[i] = np.argmax(Q[i])
        out[m, :] = 0.0
        _run_periods(Q, grid, costs[m], D[m], s, T1 - W, T1, alpha, gamma, beta, floor, T1 - W, out[m])
        same = 1.0
        for i in range(N):
            if np.argmax(Q[i]) != g0[i]:
                same = 0.0
        conv[m] = same
    return out, conv


def draw_costs(M, N, rng, base=20.0, spread=3.0, hetero=True):
    if not hetero:
        return np.full((M, N), base)
    return base + rng.uniform(-spread, spread, size=(M, N))


def assign(costs, pre, n_treat, kappa, rng, w_cost=1.0, w_pre=1.0):
    """Assignment using only information available at T0. n_treat: int or array[M].
    Score = kappa*(w_cost*z(-cost)+w_pre*z(pre profit)) + N(0,1); top n_treat in each market treated."""
    M, N = costs.shape
    n_treat = np.broadcast_to(np.asarray(n_treat), (M,))
    def z(x):
        sd = x.std(axis=1, keepdims=True)
        return (x - x.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    score = kappa * (w_cost * z(-costs) + w_pre * z(pre)) + rng.standard_normal((M, N))
    order = np.argsort(-score, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(N)[None, :].repeat(M, 0), axis=1)
    return (ranks < n_treat[:, None]).astype(np.float64)


def simulate_worlds(M, N, kappa, n_treat, alpha, seed, s=5.0, T0=2500, T1=5000, W=500,
                    gamma=0.9, beta=3e-3, floor=0.01, hetero=True, placebo=False):
    """Simulate M markets, return dict with covariates, assignment and outcomes in all worlds."""
    rng = np.random.default_rng(seed)
    costs = draw_costs(M, N, rng, hetero=hetero)
    grid = qgrid(N, s=s)
    seeds = rng.integers(1, 2**31 - 1, size=M).astype(np.int64)
    seeds_post = rng.integers(1, 2**31 - 1, size=M).astype(np.int64)
    Q0, pre = burnin(costs, grid, seeds, T0, W, alpha, gamma, beta, floor)
    D = assign(costs, pre, n_treat, kappa, rng)
    flip_idx = rng.integers(0, N, size=M)
    Dflip = D.copy()
    Dflip[np.arange(M), flip_idx] = 1.0 - Dflip[np.arange(M), flip_idx]
    s_eff = 0.0 if placebo else s
    Y0, c0 = post(Q0, costs, np.zeros((M, N)), grid, seeds_post, s_eff, T0, T1, W, alpha, gamma, beta, floor)
    Yobs, cobs = post(Q0, costs, D, grid, seeds_post, s_eff, T0, T1, W, alpha, gamma, beta, floor)
    Yall, _ = post(Q0, costs, np.ones((M, N)), grid, seeds_post, s_eff, T0, T1, W, alpha, gamma, beta, floor)
    Yflip, _ = post(Q0, costs, Dflip, grid, seeds_post, s_eff, T0, T1, W, alpha, gamma, beta, floor)
    return dict(costs=costs, pre=pre, D=D, flip_idx=flip_idx, Y0=Y0, Yobs=Yobs, Yall=Yall,
                Yflip=Yflip, conv0=c0, convobs=cobs, N=N, M=M)


def truths(sim):
    """Estimands computed from paired worlds (means over firms and markets)."""
    D, Y0, Yobs, Yall, Yflip = sim['D'], sim['Y0'], sim['Yobs'], sim['Yall'], sim['Yflip']
    M = sim['M']
    idx = np.arange(M)
    j = sim['flip_idx']
    dj = D[idx, j]
    # DE: firm j, others fixed at D_{-j}: Y_j(1,g) - Y_j(0,g)
    de_m = (Yobs[idx, j] - Yflip[idx, j]) * np.where(dj == 1, 1.0, -1.0)
    te_m = np.array([(Yobs[m][D[m] == 1] - Y0[m][D[m] == 1]).mean() if D[m].sum() > 0 else np.nan for m in range(M)])
    sp_m = np.array([(Yobs[m][D[m] == 0] - Y0[m][D[m] == 0]).mean() if (1 - D[m]).sum() > 0 else np.nan for m in range(M)])
    gte_m = (Yall - Y0).mean(axis=1)
    res = {}
    for name, v in dict(DE=de_m, TE=te_m, SP=sp_m, GTE=gte_m).items():
        v = v[~np.isnan(v)]
        res[name] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
    return res
