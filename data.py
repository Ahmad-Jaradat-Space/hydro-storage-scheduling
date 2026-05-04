"""Pull TAS1 dispatch prices via NEMOSIS, aggregate to 30-minute
trading intervals, return a tidy series indexed by SETTLEMENTDATE.

We use a longer history here than in the forecasting capstone because
the GARCH fit and the empirical price-transition matrix both want a few
years of data to settle.
"""

import os

import numpy as np
import pandas as pd
from nemosis import dynamic_data_compiler

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "cache")


def _ensure_cache():
    os.makedirs(CACHE, exist_ok=True)


def load_tas1_price(start="2022/01/01 00:00:00", end="2025/01/01 00:00:00"):
    """Return a DataFrame with 30-min TAS1 RRP for the given range."""
    _ensure_cache()
    df = dynamic_data_compiler(
        start, end, "DISPATCHPRICE", CACHE,
        filter_cols=["REGIONID"], filter_values=(["TAS1"],),
        keep_csv=False,
    )
    df = df[df["INTERVENTION"] == 0]
    df = df[["SETTLEMENTDATE", "RRP"]].copy()
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df.sort_values("SETTLEMENTDATE").set_index("SETTLEMENTDATE")
    df = df.resample("30min", label="right", closed="right").mean().dropna()
    df = df.rename(columns={"RRP": "rrp"})
    df = df.reset_index()
    return df


def add_calendar(df):
    t = df["SETTLEMENTDATE"]
    df = df.copy()
    df["hour"] = t.dt.hour
    df["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    return df


def log_return(prices, floor=1.0):
    """Stable log-return on prices that occasionally go near zero or negative.
    AEMO's market floor is -1000 $/MWh, so we shift before taking logs."""
    p = np.asarray(prices, dtype=float)
    shift = max(0.0, -p.min() + floor)
    return np.diff(np.log(p + shift))


def empirical_transition_matrix(prices, n_bins=10):
    """Markov-chain transition matrix on price-bin states.

    Bins are equal-frequency (deciles by default) on the *training*
    sample. Returns (P, edges) where P[i, j] = Pr(next bin = j | this bin = i).
    """
    edges = np.quantile(prices, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1.0
    edges[-1] += 1.0
    bins = np.clip(np.searchsorted(edges, prices, side="right") - 1, 0, n_bins - 1)
    P = np.zeros((n_bins, n_bins))
    for a, b in zip(bins[:-1], bins[1:]):
        P[a, b] += 1
    P = P / P.sum(axis=1, keepdims=True).clip(min=1)
    return P, edges, bins


def bin_centres(prices, edges):
    """Return the per-bin mean realised price (used as the reward proxy)."""
    n_bins = len(edges) - 1
    bins = np.clip(np.searchsorted(edges, prices, side="right") - 1, 0, n_bins - 1)
    centres = np.array([
        prices[bins == i].mean() if (bins == i).any()
        else 0.5 * (edges[i] + edges[i + 1])
        for i in range(n_bins)
    ])
    return centres
