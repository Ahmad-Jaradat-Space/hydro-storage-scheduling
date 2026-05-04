"""Models for the hydro-scheduling notebook.

- ARMA-GARCH price model (via the `arch` package) for residual diagnostics
  and scenario sampling.
- A small reservoir simulator with linear physics.
- A risk-neutral SDP solver on a discretised reservoir state and a
  Markov-chain price exogenous state.
- Two heuristic policies for comparison: greedy threshold dispatch and
  perfect-foresight LP (an unattainable upper bound).
"""

import numpy as np


# ===================================================================
# ARMA-GARCH wrapper
# ===================================================================
def fit_garch(returns, p=1, q=1):
    """Fit an ARMA(0,0)-GARCH(1,1) on the returns series. Returns the
    fitted model object. Use `model.simulate(...)` for scenario paths."""
    from arch import arch_model
    am = arch_model(returns, mean="ARX", lags=1, vol="GARCH", p=p, q=q,
                    dist="normal", rescale=False)
    res = am.fit(disp="off")
    return res


def simulate_paths(garch_res, last_price, last_return, n_paths, horizon, shift=0.0, rng=None):
    """Generate `n_paths` price scenarios of length `horizon` from a
    fitted ARMA-GARCH on log-returns.

    Returns an array of shape (n_paths, horizon) in price units (the
    inverse of the log/shift transform used to fit returns).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    sims = garch_res.forecast(horizon=horizon, method="simulation",
                              simulations=n_paths, reindex=False)
    # `arch` returns simulated values in `simulations.values` shape (1, n_paths, horizon)
    sim_returns = sims.simulations.values[0]   # (n_paths, horizon)
    # cumulate to log-prices, then exp and undo the shift
    log_p0 = np.log(last_price + shift)
    log_paths = log_p0 + np.cumsum(sim_returns, axis=1)
    return np.exp(log_paths) - shift


# ===================================================================
# Reservoir simulator
# ===================================================================
class Reservoir:
    """Single-reservoir hydro unit with linear water dynamics.

    Conventions (one 30-minute period = 0.5 h):
    - State `V` is reservoir level in MWh-equivalent (energy in storage).
    - Action `a` is dispatched MW for the period; energy out = 0.5 * a.
    - Inflow `i` is in MWh-equivalent per period.

    Constraints: 0 <= V <= V_max, 0 <= a <= a_max.
    Spill: any inflow that overshoots V_max is dumped without revenue.
    """

    def __init__(self, V_max=5000.0, a_max=200.0, initial_V=2500.0,
                 mean_inflow_mwh=25.0, inflow_std=8.0):
        self.V_max = float(V_max)
        self.a_max = float(a_max)
        self.initial_V = float(initial_V)
        self.mean_inflow_mwh = float(mean_inflow_mwh)
        self.inflow_std = float(inflow_std)

    def step(self, V, a, inflow):
        a = float(np.clip(a, 0, self.a_max))
        V_after = V + inflow - 0.5 * a
        spill = max(0.0, V_after - self.V_max)
        V_next = float(np.clip(V_after, 0.0, self.V_max))
        return V_next, spill

    def sample_inflow_path(self, T, rng):
        """AR(1) inflow path with mean reversion to mean_inflow_mwh."""
        phi = 0.6
        i = np.empty(T)
        x = self.mean_inflow_mwh
        for t in range(T):
            x = self.mean_inflow_mwh + phi * (x - self.mean_inflow_mwh) \
                + self.inflow_std * rng.standard_normal()
            i[t] = max(0.0, x)
        return i


# ===================================================================
# SDP on a discretised reservoir state, Markov price state
# ===================================================================
def solve_sdp(reservoir, P_trans, price_centres, T, n_V=51, n_a=21,
              mean_inflow_per_t=None, terminal_reward=0.0,
              water_value=None):
    """Backward-induction value iteration with linear V-grid interpolation
    in the continuation lookup.

    Args:
        terminal_reward : scalar OR shape-(n_V, n_p) terminal value.
        water_value     : if not None, terminal_reward is replaced with
                          `water_value * V_grid[:, None]` broadcast over
                          price bins. Models the future value of water
                          carried over the horizon — without this, the
                          SDP has no incentive to leave any water in
                          storage at t=T and dispatches it all by the
                          last few steps.

    Returns:
        V_grid       : reservoir level grid, shape (n_V,)
        a_grid       : action grid, shape (n_a,)
        value        : value function, shape (T+1, n_V, n_p)
        policy       : argmax action index, shape (T, n_V, n_p)
    """
    n_p = P_trans.shape[0]
    V_grid = np.linspace(0, reservoir.V_max, n_V)
    a_grid = np.linspace(0, reservoir.a_max, n_a)

    if mean_inflow_per_t is None:
        mean_inflow_per_t = np.full(T, reservoir.mean_inflow_mwh)

    value = np.zeros((T + 1, n_V, n_p))
    if water_value is not None:
        value[-1] = float(water_value) * V_grid[:, None] * np.ones((1, n_p))
    else:
        value[-1] = terminal_reward

    policy = np.zeros((T, n_V, n_p), dtype=np.int8)

    # action-feasibility mask: dispatching `a` MW for 0.5 h needs 0.5*a MWh
    # of water in storage. Actions that exceed (2 * V) are infeasible.
    feasible = (a_grid[None, :] <= 2 * V_grid[:, None])      # (n_V, n_a)

    dV = V_grid[1] - V_grid[0] if n_V > 1 else 1.0

    for t in range(T - 1, -1, -1):
        inflow = mean_inflow_per_t[t]
        V_after = V_grid[:, None] + inflow - 0.5 * a_grid[None, :]
        V_next = np.clip(V_after, 0.0, reservoir.V_max)

        # --- Linear interpolation over V_grid in the continuation lookup ---
        # cont has shape (n_V, n_p); we need to look it up at V_next which
        # may sit between two grid points. Floor / ceil indices + weight.
        pos = V_next / dV                                  # (n_V, n_a)
        i_lo = np.clip(np.floor(pos).astype(int), 0, n_V - 1)
        i_hi = np.clip(i_lo + 1, 0, n_V - 1)
        w_hi = np.clip(pos - i_lo, 0.0, 1.0)               # weight on i_hi
        # In-range cells where i_lo == i_hi (boundary) get w_hi=0 cleanly.

        cont = (P_trans @ value[t + 1].T).T                # (n_V, n_p)
        # Gather rows: shape (n_V, n_a, n_p)
        cont_lo = cont[i_lo]
        cont_hi = cont[i_hi]
        cont_lookup = (1.0 - w_hi[..., None]) * cont_lo + w_hi[..., None] * cont_hi

        reward = price_centres[:, None] * 0.5 * a_grid[None, :]   # (n_p, n_a)

        # Q[V, p, a] = reward[p, a] + cont_lookup[V, a, p]
        Q = reward[None, :, :] + cont_lookup.transpose(0, 2, 1)   # (n_V, n_p, n_a)
        Q = np.where(feasible[:, None, :], Q, -np.inf)
        best = Q.argmax(axis=2)
        value[t] = np.take_along_axis(Q, best[..., None], axis=2)[..., 0]
        policy[t] = best.astype(np.int8)

    return V_grid, a_grid, value, policy


def simulate_policy_sdp(reservoir, policy, V_grid, a_grid, P_edges,
                        price_path, inflow_path,
                        no_negative_dispatch=True):
    """Roll the SDP policy forward on a single realised price path.

    `no_negative_dispatch` operator override is exposed as a flag so the
    SDP, greedy, and LP can all be set consistently — the previous
    implementation hard-coded it for the SDP but not for the LP, which
    biased the comparison.

    Returns (V_history, a_history, revenue_per_t).
    """
    T = len(price_path)
    V = reservoir.initial_V
    n_V = len(V_grid); n_a = len(a_grid)
    n_p = policy.shape[2]

    Vh = np.empty(T + 1); Vh[0] = V
    ah = np.empty(T)
    rev = np.empty(T)

    for t in range(T):
        v_idx = int(np.clip(round(V / reservoir.V_max * (n_V - 1)), 0, n_V - 1))
        p_idx = int(np.clip(np.searchsorted(P_edges, price_path[t], side="right") - 1, 0, n_p - 1))
        a_idx = int(policy[t, v_idx, p_idx])
        a = a_grid[a_idx]
        a = min(a, max(0.0, V / 0.5))         # physical-feasibility cap
        if no_negative_dispatch and price_path[t] < 0:
            a = 0.0
        rev[t] = price_path[t] * 0.5 * a
        V, _ = reservoir.step(V, a, inflow_path[t])
        Vh[t + 1] = V; ah[t] = a
    return Vh, ah, rev


# ===================================================================
# Heuristic policies
# ===================================================================
def greedy_threshold(reservoir, price_path, inflow_path, threshold,
                     no_negative_dispatch=True):
    """Dispatch maximally if price > threshold and reservoir non-empty.

    `no_negative_dispatch` is moot here because threshold>=0 already gates
    out negatives, but exposed for symmetry with the SDP/LP.
    """
    T = len(price_path); V = reservoir.initial_V
    Vh = np.empty(T + 1); Vh[0] = V
    ah = np.empty(T); rev = np.empty(T)
    for t in range(T):
        if price_path[t] >= threshold and V > 0:
            a = min(reservoir.a_max, V / 0.5)
        else:
            a = 0.0
        if no_negative_dispatch and price_path[t] < 0:
            a = 0.0
        rev[t] = price_path[t] * 0.5 * a
        V, _ = reservoir.step(V, a, inflow_path[t])
        Vh[t + 1] = V; ah[t] = a
    return Vh, ah, rev


def perfect_foresight_lp(reservoir, price_path, inflow_path,
                         tight_storage=True, no_negative_dispatch=True):
    """Upper bound — solves the LP with full knowledge of price and inflow.

    Decision: dispatch d_t in [0, a_max].
    State: V_{t+1} = V_t + i_t - 0.5*d_t - spill_t,   0 <= V_t <= V_max,
    spill_t >= 0.

    `tight_storage=True` (default) enforces the actual reservoir cap by
    introducing free spill_t >= 0 alongside dispatch. The previous
    formulation relaxed the upper-bound constraint and let inflows
    accumulate without limit; with this flag the upper bound is the
    *true* perfect-foresight bound under the physics, not a relaxation.

    `no_negative_dispatch=True` enforces d_t = 0 when p_t < 0, matching
    the operator override applied to the SDP and DRL — without this, the
    LP can earn negative revenue in negative-price half-hours, biasing
    the comparison.
    """
    from scipy.optimize import linprog
    T = len(price_path)
    p = np.asarray(price_path)

    if not tight_storage:
        # Original relaxation: implicit spill, no V <= V_max constraint.
        c = -0.5 * p
        if no_negative_dispatch:
            bounds = [(0.0, 0.0) if p[t] < 0 else (0.0, reservoir.a_max)
                      for t in range(T)]
        else:
            bounds = [(0.0, reservoir.a_max)] * T
        A_ub = np.tril(np.ones((T, T))) * 0.5
        b_ub = reservoir.initial_V + np.cumsum(inflow_path)
        sol = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not sol.success:
            raise RuntimeError(f"LP failed: {sol.message}")
        d = sol.x
    else:
        # Tight version: variables are [d_0..d_{T-1}, s_0..s_{T-1}] where
        # s_t is spill at period t (>= 0). Constraints encode
        #     V_t = V_0 + sum_{u<t}(i_u - 0.5*d_u - s_u)
        #     0 <= V_t <= V_max  for t = 1..T
        # equivalently
        #     0.5*sum_{u<t} d_u + sum_{u<t} s_u <= V_0 + sum_{u<t} i_u   (V_t >= 0)
        #     -0.5*sum_{u<t} d_u - sum_{u<t} s_u <= V_max - V_0 - sum_{u<t} i_u  (V_t <= V_max)
        c = np.concatenate([-0.5 * p, np.zeros(T)])
        if no_negative_dispatch:
            d_bounds = [(0.0, 0.0) if p[t] < 0 else (0.0, reservoir.a_max)
                        for t in range(T)]
        else:
            d_bounds = [(0.0, reservoir.a_max)] * T
        s_bounds = [(0.0, None)] * T
        bounds = d_bounds + s_bounds

        L = np.tril(np.ones((T, T)))            # cumulative-sum operator
        cum_inflow = np.cumsum(inflow_path)

        # V_t >= 0  =>   0.5 * L @ d + L @ s <= V_0 + cum_inflow
        A1 = np.hstack([0.5 * L, L])
        b1 = reservoir.initial_V + cum_inflow
        # V_t <= V_max  =>  -0.5 * L @ d - L @ s <= V_max - V_0 - cum_inflow
        A2 = np.hstack([-0.5 * L, -L])
        b2 = reservoir.V_max - reservoir.initial_V - cum_inflow

        A_ub = np.vstack([A1, A2])
        b_ub = np.concatenate([b1, b2])

        sol = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not sol.success:
            raise RuntimeError(f"LP failed: {sol.message}")
        d = sol.x[:T]

    # roll forward to get the realised V trajectory
    V = reservoir.initial_V
    Vh = np.empty(T + 1); Vh[0] = V
    rev = np.empty(T)
    for t in range(T):
        rev[t] = price_path[t] * 0.5 * d[t]
        V, _ = reservoir.step(V, d[t], inflow_path[t])
        Vh[t + 1] = V
    return Vh, d, rev


# ===================================================================
# Diagnostics
# ===================================================================
def cvar(rewards, alpha=0.05):
    """Conditional value-at-risk at level alpha (left tail of total reward).
    Lower CVaR = worse worst-case."""
    cutoff = np.quantile(rewards, alpha)
    tail = rewards[rewards <= cutoff]
    return float(tail.mean()) if len(tail) > 0 else float(cutoff)


def bootstrap_metric_ci(values, fn, n_boot=500, alpha=0.05, seed=0):
    """Bootstrap 95% CI for a scalar statistic over an array of per-path
    totals. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    point = float(fn(values))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = fn(values[idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi
