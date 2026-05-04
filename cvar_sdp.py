"""Risk-averse SDP using a CVaR-Bellman equation.

We replace the risk-neutral expectation in the standard Bellman
recursion with the conditional value-at-risk at level α (the average
of the worst α-fraction of next-state values). This is implemented
via the Rockafellar-Uryasev representation:

    CVaR_α[V] = max_{η, p} { η - (1/α) Σ p_s [η - V_s]_+ }
              ≈ min_η  η + (1/α) E[ (V - η)_+ ]

Backward induction over a discretised reservoir state and a Markov
price exogenous state is identical in structure to the risk-neutral
case; the only change is that the continuation value is replaced by
its CVaR over price-transition outcomes.

For α → 1 this reduces to the risk-neutral SDP. For α → 0 it becomes
worst-case (max-min) optimisation.
"""

import numpy as np


def cvar_inner(values, probs, alpha):
    """CVaR_α of a discrete distribution over `values` with probs `probs`.
    `values` and `probs` must be 1-D arrays of the same length."""
    order = np.argsort(values)
    v = values[order]; p = probs[order]
    cum = np.cumsum(p)
    # take values up to alpha-mass
    mask_full = cum <= alpha
    if not mask_full.any():
        # alpha-mass falls inside the lowest bin
        return float(v[0])
    last = int(np.argmax(cum > alpha)) if (cum > alpha).any() else len(v) - 1
    if last == 0:
        return float(v[0])
    used_p = p[:last].copy()
    remaining = alpha - cum[last - 1]
    used_p_extra = remaining
    weighted = (used_p * v[:last]).sum() + used_p_extra * v[last]
    return float(weighted / alpha)


def solve_cvar_sdp(reservoir, P_trans, price_centres, T, alpha=0.10,
                   n_V=51, n_a=21, mean_inflow_per_t=None,
                   terminal_reward=0.0):
    """Backward induction with CVaR continuation.

    `alpha` is the CVaR level — smaller alpha is more risk-averse.
    Returns (V_grid, a_grid, value, policy) — same shapes as the
    risk-neutral version.
    """
    n_p = P_trans.shape[0]
    V_grid = np.linspace(0, reservoir.V_max, n_V)
    a_grid = np.linspace(0, reservoir.a_max, n_a)

    if mean_inflow_per_t is None:
        mean_inflow_per_t = np.full(T, reservoir.mean_inflow_mwh)

    value = np.zeros((T + 1, n_V, n_p))
    value[-1] = terminal_reward
    policy = np.zeros((T, n_V, n_p), dtype=np.int8)

    for t in range(T - 1, -1, -1):
        inflow = mean_inflow_per_t[t]
        V_after = V_grid[:, None] + inflow - 0.5 * a_grid[None, :]
        V_next = np.clip(V_after, 0.0, reservoir.V_max)
        V_next_idx = np.clip(
            np.round(V_next / reservoir.V_max * (n_V - 1)).astype(int), 0, n_V - 1
        )
        # value[t+1] shape (n_V, n_p)
        # cont_lookup[V_idx, a_idx, p_idx_next] = value[t+1, V_next_idx[V_idx, a_idx], p_idx_next]
        cont_lookup = value[t + 1][V_next_idx]  # (n_V, n_a, n_p)

        reward = price_centres[:, None] * 0.5 * a_grid[None, :]   # (n_p, n_a)

        # for each (V, p_now, a) compute CVaR_alpha over p_next using P_trans[p_now]
        new_val = np.empty((n_V, n_p))
        new_pol = np.empty((n_V, n_p), dtype=np.int8)
        for p in range(n_p):
            probs = P_trans[p]
            # cont_lookup[V_idx, a_idx, p_next] -> shape (n_V, n_a, n_p)
            # we want CVaR_alpha over the n_p axis with weights `probs`
            # vectorised: sort along last axis once per (V, a)
            cl = cont_lookup        # (n_V, n_a, n_p)
            sort_idx = np.argsort(cl, axis=2)
            cl_sorted = np.take_along_axis(cl, sort_idx, axis=2)
            p_sorted  = probs[sort_idx]
            cum = np.cumsum(p_sorted, axis=2)
            # find the bin where cum first exceeds alpha
            # everything before that bin is fully consumed; that bin partially
            below = cum <= alpha   # (n_V, n_a, n_p)
            full_mass = (p_sorted * cl_sorted * below).sum(axis=2)
            # leftover in the boundary bin
            # boundary index = first index where cum > alpha
            any_above = (~below).any(axis=2)
            first_above = np.where(any_above,
                                    np.argmax(~below, axis=2),
                                    cl.shape[2] - 1)
            # partial mass = alpha - cum[..., first_above - 1]  if first_above > 0
            pre_cum = np.zeros_like(first_above, dtype=np.float64)
            mask_pos = first_above > 0
            if mask_pos.any():
                idx = (first_above - 1).clip(min=0)
                pre_cum_arr = np.take_along_axis(cum, idx[..., None], axis=2)[..., 0]
                pre_cum = np.where(mask_pos, pre_cum_arr, 0.0)
            partial_mass = np.maximum(alpha - pre_cum, 0.0)
            partial_v = np.take_along_axis(cl_sorted, first_above[..., None], axis=2)[..., 0]
            cvar = (full_mass + partial_mass * partial_v) / alpha   # (n_V, n_a)

            # Q[V, a] = reward[p, a] + cvar[V, a]
            Q = reward[p][None, :] + cvar
            best = Q.argmax(axis=1)
            new_val[:, p] = np.take_along_axis(Q, best[:, None], axis=1)[:, 0]
            new_pol[:, p] = best.astype(np.int8)

        value[t] = new_val
        policy[t] = new_pol

    return V_grid, a_grid, value, policy
