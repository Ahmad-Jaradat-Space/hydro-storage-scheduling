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
              mean_inflow_per_t=None, terminal_reward=0.0):
    """Backward-induction value iteration.

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
    value[-1] = terminal_reward

    # next-state V index for every (V, a, t) — inflow is treated deterministically
    # at its mean to keep the state space tractable
    policy = np.zeros((T, n_V, n_p), dtype=np.int8)

    # precompute next-V indices for each (V_idx, a_idx, t)
    for t in range(T - 1, -1, -1):
        inflow = mean_inflow_per_t[t]
        # shape (n_V, n_a) of next reservoir level
        V_after = V_grid[:, None] + inflow - 0.5 * a_grid[None, :]
        V_next = np.clip(V_after, 0.0, reservoir.V_max)
        # interpolation indices on V_grid
        V_next_idx = np.clip(
            np.round(V_next / reservoir.V_max * (n_V - 1)).astype(int), 0, n_V - 1
        )

        # expected continuation value over price transitions
        # shape (n_p, n_V, n_a)
        # = sum_p' P[p, p'] * value[t+1, V_next_idx, p']
        # we compute as P @ V_{t+1}[V_next_idx] via tensor manipulation
        cont = (
            P_trans @ value[t + 1].T   # shape (n_p, n_V)
        ).T  # shape (n_V, n_p)
        # cont_lookup[V_idx, a_idx, p_idx] = cont[V_next_idx[V_idx, a_idx], p_idx]
        cont_lookup = cont[V_next_idx]   # shape (n_V, n_a, n_p)

        # immediate reward shape (n_p, n_a) = price * 0.5 * a
        reward = price_centres[:, None] * 0.5 * a_grid[None, :]

        # Q[V, p, a] = reward[p, a] + cont_lookup[V, a, p]
        Q = reward[None, :, :] + cont_lookup.transpose(0, 2, 1)
        best = Q.argmax(axis=2)
        value[t] = np.take_along_axis(Q, best[..., None], axis=2)[..., 0]
        policy[t] = best.astype(np.int8)

    return V_grid, a_grid, value, policy


def simulate_policy_sdp(reservoir, policy, V_grid, a_grid, P_edges,
                        price_path, inflow_path):
    """Roll the SDP policy forward on a single realised price path.
    Returns (V_history, a_history, revenue_per_t)."""
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
        rev[t] = price_path[t] * 0.5 * a
        V, _ = reservoir.step(V, a, inflow_path[t])
        Vh[t + 1] = V; ah[t] = a
    return Vh, ah, rev


# ===================================================================
# Heuristic policies
# ===================================================================
def greedy_threshold(reservoir, price_path, inflow_path, threshold):
    """Dispatch maximally if price > threshold and reservoir non-empty."""
    T = len(price_path); V = reservoir.initial_V
    Vh = np.empty(T + 1); Vh[0] = V
    ah = np.empty(T); rev = np.empty(T)
    for t in range(T):
        if price_path[t] >= threshold and V > 0:
            a = min(reservoir.a_max, V / 0.5)
        else:
            a = 0.0
        rev[t] = price_path[t] * 0.5 * a
        V, _ = reservoir.step(V, a, inflow_path[t])
        Vh[t + 1] = V; ah[t] = a
    return Vh, ah, rev


def perfect_foresight_lp(reservoir, price_path, inflow_path):
    """Upper bound — solves the LP with full knowledge of price and inflow.
    Decision: dispatch d_t in [0, a_max].
    State: V_{t+1} = V_t + i_t - 0.5*d_t,   0 <= V_t <= V_max.
    Maximise sum p_t * 0.5 * d_t.
    """
    from scipy.optimize import linprog
    T = len(price_path)
    # decision vector x = [d_0 .. d_{T-1}]   shape (T,)
    # objective: minimise -sum p_t * 0.5 * d_t
    c = -0.5 * np.asarray(price_path)

    # bounds 0 <= d_t <= a_max
    bounds = [(0.0, reservoir.a_max)] * T

    # state inequalities: 0 <= V_0 + sum_{s<=t}(i_s - 0.5*d_s) <= V_max
    # i.e. -V_max + V_0 + sum(i_s) <= sum 0.5*d_s <= V_0 + sum(i_s) - 0
    # Build the cumulative coefficient matrix
    A_ub_lo = np.tril(np.ones((T, T))) * 0.5    # lower bound: V >= 0
    A_ub_hi = -A_ub_lo                          # upper bound: V <= V_max
    cum_inflow = np.cumsum(inflow_path)
    b_ub_lo = reservoir.initial_V + cum_inflow         # 0.5*sum(d) <= V_0 + sum(i)
    b_ub_hi = reservoir.V_max - reservoir.initial_V - cum_inflow  # -0.5*sum(d) <= V_max - V_0 - sum(i)

    A_ub = np.vstack([A_ub_lo, A_ub_hi])
    b_ub = np.concatenate([b_ub_lo, b_ub_hi])

    sol = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not sol.success:
        raise RuntimeError(f"LP failed: {sol.message}")
    d = sol.x

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
