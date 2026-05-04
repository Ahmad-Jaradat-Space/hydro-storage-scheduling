"""Markov-switching GARCH — two-regime ARCH(1) for tractability.

Two regimes (low-volatility coupled vs high-volatility detached). The
hidden regime evolves as a discrete Markov chain; conditional on the
regime, returns are normal with regime-specific mean and ARCH(1)
variance. Inference via the standard Hamilton filter (forward
filter + backward smoother), implemented in numpy for speed and
clarity.

Why ARCH(1) rather than GARCH(1,1) for this section: the ARCH(1)
variance depends only on the previous return squared, which keeps
the latent state finite-dimensional (regime alone) and the Hamilton
filter exact. Full MS-GARCH with persistence is a celebrated open
problem (Klaassen 2002; Haas et al. 2004) and the right
demonstration in this repo is the regime structure itself, not the
heroics of solving the persistence approximation.
"""

import numpy as np
from scipy.optimize import minimize


def _gauss_logpdf(x, mu, var):
    var = np.clip(var, 1e-9, None)
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu) ** 2 / var


def hamilton_filter(returns, params):
    """Forward Hamilton filter giving filtered regime probabilities."""
    p11, p22, mu1, mu2, omega1, omega2, alpha1, alpha2 = params
    P = np.array([[p11, 1 - p11], [1 - p22, p22]])

    T = len(returns)
    pi = np.array([1.0 - p11, 1.0 - p22])
    pi = pi / pi.sum()
    filt = np.zeros((T, 2))

    h_prev = np.array([omega1 / (1 - alpha1 + 1e-6),
                       omega2 / (1 - alpha2 + 1e-6)])
    r_prev = 0.0
    log_lik = 0.0
    for t in range(T):
        h = np.array([omega1 + alpha1 * r_prev ** 2,
                      omega2 + alpha2 * r_prev ** 2])
        # state-conditional likelihood
        ll = _gauss_logpdf(returns[t], np.array([mu1, mu2]), h)
        # prior = pi @ P
        prior = pi @ P
        joint = prior * np.exp(ll - ll.max())
        denom = joint.sum() + 1e-12
        post = joint / denom
        filt[t] = post
        log_lik += np.log(denom) + ll.max()
        pi = post
        r_prev = returns[t]
        h_prev = h
    return filt, log_lik


def neg_log_lik(theta, returns):
    p11 = 1 / (1 + np.exp(-theta[0]))   # logits → probs
    p22 = 1 / (1 + np.exp(-theta[1]))
    mu1, mu2 = theta[2], theta[3]
    omega1 = np.exp(theta[4]); omega2 = np.exp(theta[5])
    alpha1 = 1 / (1 + np.exp(-theta[6]))
    alpha2 = 1 / (1 + np.exp(-theta[7]))
    _, ll = hamilton_filter(returns, (p11, p22, mu1, mu2,
                                       omega1, omega2, alpha1, alpha2))
    return -ll


def fit(returns, n_starts=4, seed=0):
    best = None; best_x = None
    rng = np.random.default_rng(seed)
    starts = [
        np.array([2.0, 2.0, 0.0, 0.0, np.log(0.001), np.log(0.01), -1.0, -0.5]),
        np.array([3.0, 3.0, -0.05, 0.05, np.log(0.0005), np.log(0.02), -2.0, -1.0]),
    ] + [rng.standard_normal(8) * 0.5 for _ in range(n_starts - 2)]
    for x0 in starts:
        try:
            res = minimize(neg_log_lik, x0, args=(returns,),
                           method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-3})
            if best is None or res.fun < best:
                best = res.fun; best_x = res.x
        except Exception:
            continue
    theta = best_x
    p11 = 1 / (1 + np.exp(-theta[0]))
    p22 = 1 / (1 + np.exp(-theta[1]))
    params = (p11, p22, theta[2], theta[3],
              float(np.exp(theta[4])), float(np.exp(theta[5])),
              1 / (1 + np.exp(-theta[6])), 1 / (1 + np.exp(-theta[7])))
    filt, ll = hamilton_filter(returns, params)
    return params, filt, ll
