"""Plot helpers for the hydro-scheduling notebook."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ACCENT = "#2c5f8d"
WARN = "#c44e52"
CONTEXT = "#7f7f7f"
GOOD = "#3a8c5f"


def apply_style():
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


def caption(fig, text):
    fig.text(0.5, -0.04, text, ha="center", va="top",
             fontsize=9, color="#555", style="italic", wrap=True)


def price_overview(df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(df["SETTLEMENTDATE"], df["rrp"], color=ACCENT, lw=0.6, alpha=0.8)
    ax.set_ylabel("RRP ($/MWh)")
    ax.set_xlabel("")
    return ax


def acf_pacf(returns, ax_acf=None, ax_pacf=None, lags=48):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    if ax_acf is None:
        _, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(13, 3.5))
    plot_acf(returns, lags=lags, ax=ax_acf, color=ACCENT)
    ax_acf.set_title("ACF of log-returns")
    plot_pacf(returns, lags=lags, ax=ax_pacf, color=ACCENT, method="ywm")
    ax_pacf.set_title("PACF of log-returns")
    return ax_acf, ax_pacf


def vol_clustering(returns, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(np.arange(len(returns)), returns ** 2, color=WARN, lw=0.4, alpha=0.7)
    ax.set_xlabel("trading interval index")
    ax.set_ylabel("squared log-return")
    return ax


def scenarios_fan(scenarios, real=None, ax=None, n_show=200):
    """Plot a sample of simulated price paths with a quantile band."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    n = scenarios.shape[0]
    sel = np.random.default_rng(0).choice(n, min(n_show, n), replace=False)
    for s in sel:
        ax.plot(scenarios[s], color=ACCENT, lw=0.4, alpha=0.08)
    q10 = np.quantile(scenarios, 0.10, axis=0)
    q50 = np.quantile(scenarios, 0.50, axis=0)
    q90 = np.quantile(scenarios, 0.90, axis=0)
    t = np.arange(scenarios.shape[1])
    ax.fill_between(t, q10, q90, color=ACCENT, alpha=0.2,
                    label="10–90% band")
    ax.plot(t, q50, color=ACCENT, lw=1.2, label="median")
    if real is not None:
        ax.plot(t, real, color="black", lw=1.2, label="realised")
    ax.set_xlabel("half-hour from now")
    ax.set_ylabel("price ($/MWh)")
    ax.legend()
    return ax


def garch_residual_diagnostics(std_resid, axes=None):
    """Histogram + QQ of standardised residuals, side by side."""
    import matplotlib.pyplot as plt
    from scipy import stats
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(std_resid, bins=80, density=True, alpha=0.7, color=ACCENT)
    xs = np.linspace(-5, 5, 200)
    axes[0].plot(xs, stats.norm.pdf(xs), color="black", lw=1.2, label="N(0,1)")
    axes[0].set_xlabel("standardised residual")
    axes[0].set_ylabel("density")
    axes[0].set_title("Residuals vs N(0,1)")
    axes[0].legend()
    stats.probplot(std_resid, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_color(ACCENT)
    axes[1].get_lines()[0].set_markersize(3)
    axes[1].get_lines()[1].set_color(WARN)
    axes[1].set_title("Normal QQ")
    return axes


def trajectory_overlay(price_path, all_traj, V_max, a_max, axes=None):
    """Three-panel overlay for multiple policies on the same realised path."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    palette = {"perfect-foresight LP": "#7e57c2",
               "SDP": ACCENT,
               "greedy threshold": GOOD}
    T = len(price_path); ts = np.arange(T)
    axes[0].plot(ts, price_path, color="black", lw=1.0)
    axes[0].set_ylabel("price ($/MWh)")
    axes[0].set_title("Realised TAS1 price (one out-of-sample day)")
    for name, (Vh, ah, _) in all_traj.items():
        col = palette.get(name, ACCENT)
        axes[1].step(ts, ah, where="post", color=col, lw=1.0,
                     alpha=0.85, label=name)
        axes[2].plot(np.arange(len(Vh)), Vh, color=col, lw=1.0,
                     alpha=0.85, label=name)
    axes[1].set_ylim(-5, a_max + 5)
    axes[1].set_ylabel("dispatch (MW)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_title("Dispatch — when each policy releases water")
    axes[2].set_ylim(-50, V_max + 50)
    axes[2].set_ylabel("reservoir (MWh)")
    axes[2].set_xlabel("half-hour from horizon start")
    axes[2].set_title("Reservoir level")
    return axes


def policy_heatmap_grid(policy, V_grid, t_list, titles, axes=None):
    """Side-by-side policy heatmaps at several timesteps."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    n = len(t_list)
    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), sharey=True)
    for ax, t, title in zip(axes, t_list, titles):
        sns.heatmap(policy[t].astype(float), cmap="viridis",
                    cbar=(t == t_list[-1]),
                    cbar_kws={"label": "action index"} if t == t_list[-1] else None,
                    xticklabels=False, yticklabels=False, ax=ax)
        ax.set_xlabel("price decile (low → high)")
        ax.set_title(title)
    axes[0].set_ylabel("reservoir level (empty → full)")
    return axes


def policy_heatmap(policy, V_grid, edges, t=0, ax=None):
    """Heatmap of optimal action vs (reservoir level, price decile) at time t."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    # policy shape (T, n_V, n_p) -> select t
    img = policy[t].astype(float)  # n_V x n_p
    sns.heatmap(img, cmap="viridis", cbar_kws={"label": "action index"},
                ax=ax, xticklabels=False, yticklabels=False)
    ax.set_xlabel("price decile (low → high)")
    ax.set_ylabel("reservoir level (empty → full)")
    return ax


def trajectory(ts, Vh, ah, rev, price_path, V_max, a_max, ax=None):
    """Three-panel: price, dispatch, reservoir level."""
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(ts, price_path, color=ACCENT, lw=1.0)
    ax[0].set_ylabel("price ($/MWh)")
    ax[1].step(ts, ah, color=GOOD, where="post", lw=1.0)
    ax[1].set_ylim(-5, a_max + 5)
    ax[1].set_ylabel("dispatch (MW)")
    ax[2].plot(np.arange(len(Vh)), Vh, color=WARN, lw=1.0)
    ax[2].set_ylim(-50, V_max + 50)
    ax[2].set_ylabel("reservoir (MWh)")
    ax[2].set_xlabel("half-hour from horizon start")
    return ax


def revenue_distribution(rev_dict, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    palette = [ACCENT, WARN, GOOD, "#7e57c2"]
    for (name, rev), col in zip(rev_dict.items(), palette):
        ax.hist(rev.sum(axis=1), bins=30, alpha=0.5,
                color=col, label=f"{name} (mean {rev.sum(axis=1).mean():,.0f})")
    ax.set_xlabel("total revenue over horizon ($)")
    ax.set_ylabel("count of price paths")
    ax.legend()
    return ax


def policy_comparison_bar(summary, ax=None):
    """Horizontal bar chart of mean revenue per policy with CVaR overlay."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    df = summary.sort_values("mean")
    y = np.arange(len(df))
    h = 0.35
    ax.barh(y - h / 2, df["mean"], height=h, color=ACCENT, label="mean revenue")
    ax.barh(y + h / 2, df["cvar5"], height=h, color=WARN, label="CVaR @ 5% (worst-case avg)")
    for i, (m, c) in enumerate(zip(df["mean"], df["cvar5"])):
        ax.text(m, i - h / 2, f" {m:,.0f}", va="center", fontsize=8)
        ax.text(c, i + h / 2, f" {c:,.0f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["policy"])
    ax.set_xlabel("revenue ($)")
    ax.legend(loc="lower right")
    return ax
