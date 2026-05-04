"""Bayesian ARMA-GARCH(1,1) in numpyro.

Replaces the MLE fit from `arch_model` with a fully Bayesian posterior
over the conditional-variance parameters. Posterior predictive
scenarios are then used to drive the SDP in §5.

Model (returns r_t = AR(1) mean + GARCH(1,1) noise):
    h_t = ω + α (r_{t-1} - μ - φ r_{t-2})^2 + β h_{t-1}
    r_t = μ + φ r_{t-1} + sqrt(h_t) · z_t,    z_t ~ N(0, 1)

Priors are weakly informative.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np


def model(r):
    T = r.shape[0]
    mu = numpyro.sample("mu", dist.Normal(0.0, 0.05))
    phi = numpyro.sample("phi", dist.Normal(0.0, 0.5))
    omega = numpyro.sample("omega", dist.HalfNormal(0.05))
    alpha = numpyro.sample("alpha", dist.Beta(2.0, 5.0))
    beta = numpyro.sample("beta", dist.Beta(5.0, 2.0))

    # numerical guard so omega + alpha + beta < 1 (stationarity)
    numpyro.factor("stationarity",
                   jnp.where(alpha + beta < 0.999, 0.0, -1e9))

    h0 = omega / jnp.maximum(1 - alpha - beta, 1e-3)

    def step(state, t):
        r_prev, r_prev2, h_prev = state
        # Correct ARMA(1,0)-GARCH(1,1) innovation:
        # eps_{t-1} = r_{t-1} - mu - phi * r_{t-2}
        eps_prev = r_prev - mu - phi * r_prev2
        h_t = omega + alpha * eps_prev ** 2 + beta * h_prev
        return (r[t], r_prev, h_t), h_t

    # carry both the most-recent return and the one before so eps_{t-1}
    # uses the correct r_{t-2} term, not r_{t-1} as a placeholder.
    init = (r[0], jnp.float32(0.0), h0)
    _, h_seq = jax.lax.scan(step, init, jnp.arange(1, T))
    h = jnp.concatenate([jnp.array([h0]), h_seq])

    mean = mu + phi * jnp.concatenate([jnp.array([0.0]), r[:-1]])
    numpyro.sample("r_obs", dist.Normal(mean, jnp.sqrt(jnp.clip(h, 1e-8, 100.0))),
                   obs=r)


def fit(returns, n_warmup=300, n_samples=300, sub_every=4, seed=0):
    from numpyro.infer import NUTS, MCMC
    r = jnp.asarray(returns[::sub_every].astype(np.float32))
    rng = jax.random.PRNGKey(seed)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=1, progress_bar=False)
    mcmc.run(rng, r=r)
    return mcmc, mcmc.get_samples()


def simulate_paths(samples, last_price, last_return, n_paths, horizon,
                   shift=0.0, rng=None):
    """Generate price scenarios using posterior parameter samples."""
    if rng is None:
        rng = np.random.default_rng(0)
    keys_to_check = ("mu", "phi", "omega", "alpha", "beta")
    n_post = len(samples[keys_to_check[0]])
    # for each path, draw one parameter set from the posterior
    pick = rng.integers(0, n_post, size=n_paths)
    mu = np.array(samples["mu"])[pick]
    phi = np.array(samples["phi"])[pick]
    omega = np.array(samples["omega"])[pick]
    alpha = np.array(samples["alpha"])[pick]
    beta = np.array(samples["beta"])[pick]

    # initial unconditional variance
    h_prev = omega / np.maximum(1 - alpha - beta, 1e-3)
    r_prev = np.full(n_paths, float(last_return), dtype=np.float64)
    # r_{t-2} starts at the unconditional mean (0 in log-returns space)
    r_prev2 = np.zeros(n_paths, dtype=np.float64)

    log_p = np.full(n_paths, float(np.log(last_price + shift)), dtype=np.float64)
    out = np.empty((n_paths, horizon), dtype=np.float64)
    for t in range(horizon):
        # mirrors the corrected model: eps_{t-1} = r_{t-1} - mu - phi * r_{t-2}
        eps_prev = r_prev - mu - phi * r_prev2
        h = omega + alpha * eps_prev ** 2 + beta * h_prev
        h = np.clip(h, 1e-8, 100.0)
        z = rng.standard_normal(n_paths)
        r_t = mu + phi * r_prev + np.sqrt(h) * z
        log_p = log_p + r_t
        out[:, t] = np.exp(log_p) - shift
        r_prev2 = r_prev
        r_prev = r_t
        h_prev = h
    # AEMO market price floor / ceiling. The Bayesian GARCH on log-returns
    # can drift outside these in long simulations; clipping reflects the
    # actual market design.
    return np.clip(out, -1000.0, 17500.0)
