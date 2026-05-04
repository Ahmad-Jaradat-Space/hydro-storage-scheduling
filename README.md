# Hydro Storage Scheduling under Price Uncertainty (TAS1)

Tasmania's electricity system is ~80% hydro, and the reservoirs are not just generation — they are *energy storage*. So Hydro Tasmania's daily problem is the canonical one in energy economics: when do you let the water through, and when do you hold it back? This repo builds a small but honest version of that problem on real TAS1 dispatch prices.

The notebook puts three claims on trial:

1. The textbook stochastic-dynamic-programming answer, given a halfway-decent probabilistic price model, beats a sensible threshold heuristic — both on average and in the worst-case tail.
2. The remaining gap to a perfect-foresight upper bound is essentially the *value of better price forecasts*; it would shrink with a richer mean model.
3. The SDP's optimal policy reproduces the qualitative rule of thumb a hydro operator would write down by hand — the value of doing it formally is in being precise about *where* the dispatch thresholds sit, not in inventing a new rule.

## How the notebook is laid out

The notebook reads as a short paper with five sections:

1. **Introduction** — the problem and why Tasmania is the right setting.
2. **Data** — TAS1 prices pulled via [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS) for 2022–2024, autocorrelation and volatility-clustering diagnostics that motivate the model choice.
3. **Methods** — six progressively richer chapters: (1) wavelet spectral characterisation, (2) Bayesian ARMA-GARCH in numpyro with full posterior over the volatility parameters, (3) Markov-switching ARCH with a Hamilton filter for explicit regime inference, (4) reservoir simulator with linear physics, (5) risk-neutral SDP via backward induction on a discretised state, (6) risk-averse CVaR-Bellman SDP with a sweep over risk-aversion levels.
4. **Results** — Monte Carlo revenue distributions across 200 paths sampled from the Bayesian posterior predictive, mean–CVaR risk/return frontier, plus a from-scratch PyTorch PPO benchmark trained on the reservoir as a Gym-like environment.
5. **Conclusion** — answers to the three claims and what would change before this could become a real scheduler.

Every plot is read out loud: a one-line setup before the cell, a finding-style title, and a 2–4 sentence takeaway after — *what to look at, what it means, what it indicates next.*

## Running it

Tested on macOS with Python 3.12.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The first run downloads about 36 months of TAS1 dispatch CSVs into `data/cache/` (gitignored) — slow the first time, instant after that thanks to NEMOSIS's feather cache. The committed `notebook.ipynb` is already executed, so GitHub renders all outputs and plots inline without needing to run anything.

## What's where

- `notebook.ipynb` — the whole story, runs top to bottom
- `data.py` — NEMOSIS price pulls, 30-min aggregation, empirical Markov transition matrix on price deciles
- `models.py` — reservoir simulator, risk-neutral SDP solver, greedy threshold and perfect-foresight LP benchmarks, CVaR helper
- `spectral.py` — continuous Morlet wavelet + STFT
- `bayes_garch.py` — Bayesian ARMA-GARCH(1,1) in numpyro with posterior predictive scenarios
- `ms_garch.py` — Markov-switching ARCH(1) with Hamilton filter
- `cvar_sdp.py` — risk-averse CVaR-Bellman SDP solver
- `drl.py` — Gym-like reservoir env + from-scratch PPO actor-critic in PyTorch
- `plots.py` — small matplotlib helpers used by the notebook
