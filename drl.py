"""Deep RL benchmark for the hydro scheduling problem.

A small custom Gym-like environment wrapping the reservoir model,
plus a from-scratch PPO agent in PyTorch. We avoid `stable-baselines3`
to keep the dependency footprint small and the implementation
inspectable. The agent acts continuously (clipped Gaussian over
dispatch MW) and observes (V, current_price_normalised, time_in_episode).

For a 48-step horizon and 50k steps of training, this fits in a few
minutes on CPU. The point is to demonstrate the *technique*, not to
exhaustively beat the SDP — it is well known that DRL only beats
classical DP when the state space is high-dimensional or the
dynamics resist discretisation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HydroEnv:
    def __init__(self, reservoir, scenarios, horizon, rng_seed=0):
        self.reservoir = reservoir
        self.scenarios = scenarios.astype(np.float32)
        self.horizon = horizon
        self.rng = np.random.default_rng(rng_seed)
        # Use the 99.9th percentile + a log transform for normalisation so
        # the agent actually sees the heavy right tail. The previous
        # 99th-percentile + linear normalisation clipped the agent's view
        # of the spike events the storage policy exists to capture.
        self.price_norm_ref = float(np.percentile(scenarios, 99.9) + 1.0)
        self.reset()

    def _norm_price(self, price):
        # Symmetric log: positive prices saturate slowly, negatives go negative.
        return float(np.sign(price) * np.log1p(np.abs(price)) /
                     np.log1p(self.price_norm_ref))

    def reset(self):
        self.t = 0
        self.V = float(self.reservoir.initial_V)
        self.path_idx = int(self.rng.integers(0, len(self.scenarios)))
        self.inflow = self.reservoir.sample_inflow_path(self.horizon, self.rng)
        return self._obs()

    def _obs(self):
        price = self.scenarios[self.path_idx, self.t]
        return np.array([
            self.V / self.reservoir.V_max,
            self._norm_price(price),
            self.t / self.horizon,
        ], dtype=np.float32)

    def step(self, a_norm):
        a = float(np.clip(a_norm, 0.0, 1.0)) * self.reservoir.a_max
        a = min(a, max(0.0, self.V / 0.5))
        price = float(self.scenarios[self.path_idx, self.t])
        if price < 0:                          # operator override
            a = 0.0
        reward = price * 0.5 * a
        self.V, _ = self.reservoir.step(self.V, a, self.inflow[self.t])
        self.t += 1
        done = self.t >= self.horizon
        # if we've stepped past the last interval, return a dummy obs
        # (the agent shouldn't act on it; `done` is true)
        if done:
            obs = np.array([self.V / self.reservoir.V_max, 0.0, 1.0],
                           dtype=np.float32)
            return obs, reward, True
        return self._obs(), reward, done


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        h = self.shared(obs)
        mean = torch.sigmoid(self.actor_mean(h)).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        if squeeze:
            return mean.squeeze(0), std.squeeze(0), value.squeeze(0)
        return mean, std, value


def collect_trajectory(env, ac, horizon):
    obs_list, act_list, logp_list, rew_list, val_list = [], [], [], [], []
    obs = env.reset()
    for _ in range(horizon):
        ot = torch.from_numpy(obs).float()
        with torch.no_grad():
            mean, std, value = ac(ot)
            dist = torch.distributions.Normal(mean, std)
            a = dist.sample().clamp(0.0, 1.0)
            logp = dist.log_prob(a)
        nxt, rew, done = env.step(float(a.item()))
        obs_list.append(obs); act_list.append(float(a.item()))
        logp_list.append(float(logp.item())); rew_list.append(float(rew))
        val_list.append(float(value.item()))
        obs = nxt
        if done:
            break
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32), \
           np.array(logp_list, dtype=np.float32), np.array(rew_list, dtype=np.float32), \
           np.array(val_list, dtype=np.float32)


def gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    next_v = 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_v - values[t]
        adv[t] = last = delta + gamma * lam * last
        next_v = values[t]
    returns = adv + values
    return adv, returns


def train_ppo(env, total_steps=50_000, horizon=48,
              clip=0.2, epochs=4, batch_size=64, lr=3e-4,
              entropy_coef=0.01, seed=0):
    torch.manual_seed(seed)
    obs_dim = 3
    ac = ActorCritic(obs_dim, hidden=64)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)

    reward_history = []
    steps_done = 0
    while steps_done < total_steps:
        obs_b, act_b, logp_b, rew_b, val_b = collect_trajectory(env, ac, horizon)
        ep_return = float(rew_b.sum())
        reward_history.append(ep_return)
        adv, ret = gae(rew_b, val_b)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.from_numpy(obs_b)
        act_t = torch.from_numpy(act_b)
        logp_t = torch.from_numpy(logp_b)
        adv_t = torch.from_numpy(adv)
        ret_t = torch.from_numpy(ret)

        n = len(obs_t)
        for _ in range(epochs):
            idx = torch.randperm(n)
            for s in range(0, n, batch_size):
                b = idx[s:s + batch_size]
                mean, std, value = ac(obs_t[b])
                dist = torch.distributions.Normal(mean, std)
                new_logp = dist.log_prob(act_t[b])
                ratio = torch.exp(new_logp - logp_t[b])
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_t[b])
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
                opt.zero_grad(); loss.backward(); opt.step()

        steps_done += len(obs_b)
    return ac, reward_history


def rollout_policy(env, ac, horizon):
    obs = env.reset()
    Vh = [env.V]; ah = []; rew = []
    for _ in range(horizon):
        ot = torch.from_numpy(obs).float()
        with torch.no_grad():
            mean, _, _ = ac(ot)
        a = float(mean.item())
        nxt, r, done = env.step(a)
        rew.append(r); ah.append(a * env.reservoir.a_max); Vh.append(env.V)
        obs = nxt
        if done:
            break
    return np.array(Vh), np.array(ah), np.array(rew)
