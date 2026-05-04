"""Plot helpers for the hydro-scheduling notebook.

Production-grade design system (paper-toned, coherent palette, finding-style
titles) is shared with the other capstone repos. Domain-specific helpers
below stay scientific."""

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Design palette
PRIMARY = "#0E4F5F"
ACCENT  = "#D88C4A"
GOOD    = "#5C9D7E"
WARN    = "#B0413E"
MUTED   = "#7A8C99"
INK     = "#1A1A1A"
PAPER   = "#FAFAF7"
STORAGE = "#5E548E"
RAIN    = "#3F7CAC"
CONTEXT = MUTED


def apply_style():
    sns.set_theme(style="white", context="notebook")
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 140,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12.5,
        "axes.titlepad": 12,
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#BFC4CA",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": "sans-serif",
        "font.size": 10.5,
    })
    try:
        mpl.rcParams["text.parse_math"] = False
    except KeyError:
        pass


def caption(fig, text):
    fig.text(0.5, -0.04, text, ha="center", va="top",
             fontsize=9, color="#555", style="italic", wrap=True)


def annotate_event(ax, x, y, text, dx=20, dy=20, color=INK):
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=9, color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec=color, lw=0.6, alpha=0.95),
    )


def kpi_card(ax, value, label, sub=None, color=PRIMARY):
    ax.axis("off")
    ax.text(0.5, 0.62, value, ha="center", va="center",
            fontsize=22, color=color, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.30, label, ha="center", va="center",
            fontsize=10, color=INK, transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.12, sub, ha="center", va="center",
                fontsize=8.5, color=MUTED, transform=ax.transAxes, style="italic")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    fill=False, ec="#D8DCE2", lw=1.0))


def kpi_banner(values):
    fig, axes = plt.subplots(1, len(values), figsize=(2.8 * len(values), 1.8))
    if len(values) == 1:
        axes = [axes]
    palette = [PRIMARY, ACCENT, WARN, GOOD, STORAGE]
    for ax, v, color in zip(axes, values, palette):
        kpi_card(ax, *v, color=color)
    plt.tight_layout()
    return fig


def business_summary(rows, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 1))
    ax.axis("off")
    n = len(rows)
    for i, (lhs, rhs) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        ax.text(0.02, y, lhs, transform=ax.transAxes,
                fontsize=10.5, color=INK, fontweight="bold", va="center")
        ax.text(0.34, y, rhs, transform=ax.transAxes,
                fontsize=10, color=INK, va="center")
        ax.plot([0.01, 0.99], [1 - i / n, 1 - i / n],
                color="#E0E4EA", lw=0.6, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    return ax


def dispatch_decision_rule(price_grid, action_for_levels, V_max, ax=None):
    """Optimal action vs price for three reservoir-level slices.

    `action_for_levels` is dict {label: action_array_per_price_decile}."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    palette = {"empty (10%)": WARN, "half (50%)": PRIMARY, "full (90%)": GOOD}
    for lbl, acts in action_for_levels.items():
        col = palette.get(lbl, PRIMARY)
        ax.step(price_grid, acts, where="post", color=col, lw=1.8,
                label=lbl, alpha=0.95)
    ax.set_xlabel("Price decile (low → high)")
    ax.set_ylabel("Optimal dispatch (MW)")
    ax.set_title("SDP decision rule: when does each storage level release?",
                 color=INK)
    ax.legend(loc="upper left", title="Reservoir level")
    return ax


def value_function_surface(V_grid, p_grid, V_values, t_label, ax=None):
    """Filled contour of the SDP value function at a single timestep."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    cs = ax.contourf(p_grid, V_grid, V_values, levels=18, cmap="viridis")
    ax.set_xlabel("Price decile")
    ax.set_ylabel("Reservoir level (MWh)")
    plt.colorbar(cs, ax=ax, label="V(t, V, p)  $")
    ax.set_title(f"Value function at {t_label}: red corners = release fast, dark = hold",
                 color=INK)
    return ax


def spillage_timeline(t, inflow, dispatch, level, V_max, ax=None):
    """Inflow / dispatch / level / spill stacked time-series for the perfect-LP."""
    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    ax[0].fill_between(t, 0, inflow, color=GOOD, alpha=0.55, lw=0)
    ax[0].set_ylabel("Inflow (MWh / 30min)")
    ax[1].step(t, dispatch, where="post", color=PRIMARY, lw=1.0)
    ax[1].set_ylabel("Dispatch (MW)")
    ax[2].plot(t, level, color=STORAGE, lw=1.4)
    ax[2].axhline(V_max, color=WARN, ls="--", lw=1.0, label=f"cap {V_max}")
    ax[2].fill_between(t, level, V_max, where=level >= V_max - 1, color=WARN, alpha=0.35,
                       label="spilling")
    ax[2].set_ylabel("Level (MWh)")
    ax[2].set_xlabel("Half-hour")
    ax[2].legend(loc="upper right")
    ax[0].set_title("Perfect-foresight LP: where the cap binds, water spills",
                    color=INK)
    return ax


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


def scalogram(coeffs, periods_hours, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))
    power = np.log10(np.abs(coeffs) ** 2 + 1e-9)
    extent = [0, coeffs.shape[1], periods_hours[0], periods_hours[-1]]
    im = ax.imshow(power, aspect="auto", origin="lower",
                   cmap="magma", extent=extent)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 6, 12, 24, 24 * 7])
    ax.get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} h")
    )
    ax.set_ylabel("period"); ax.set_xlabel("time index (30-min)")
    plt.colorbar(im, ax=ax, label="log10 power")
    return ax


def posterior_pair(samples, names, ax=None):
    """Trace + density mini-grid for the named parameters."""
    import matplotlib.pyplot as plt
    n = len(names)
    if ax is None:
        fig, ax = plt.subplots(n, 2, figsize=(11, 1.8 * n))
    for i, name in enumerate(names):
        s = np.asarray(samples[name])
        ax[i, 0].plot(s, color=ACCENT, lw=0.6)
        ax[i, 0].set_ylabel(name)
        ax[i, 1].hist(s, bins=40, color=ACCENT, alpha=0.85)
        ax[i, 1].set_yticks([])
        ax[i, 1].axvline(np.median(s), color=WARN, lw=1.0)
    ax[0, 0].set_title("trace")
    ax[0, 1].set_title("posterior density")
    ax[-1, 0].set_xlabel("MCMC iteration")
    ax[-1, 1].set_xlabel("value")
    return ax


def regime_ribbon(prices, regime_prob_high, ax=None):
    """Price series with a coloured ribbon below showing regime probability."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(13, 4),
                               gridspec_kw={"height_ratios": [3, 1]},
                               sharex=True)
    ax[0].plot(prices, color=ACCENT, lw=0.6)
    ax[0].set_ylabel("RRP ($/MWh)")
    ax[0].set_title("TAS1 prices, with inferred regime probability below")
    ax[1].fill_between(np.arange(len(regime_prob_high)),
                       0, regime_prob_high, color=WARN, alpha=0.7)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("P(high-vol)")
    ax[1].set_xlabel("trading interval index")
    return ax


def cvar_frontier(frontier, ax=None):
    """Pareto plot of (mean revenue, CVaR_5%) across CVaR-SDP risk levels."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    means = [r["mean"] for r in frontier]
    cvars = [r["cvar5"] for r in frontier]
    labels = [r["label"] for r in frontier]
    ax.plot(cvars, means, "-o", color=ACCENT, lw=1.5, ms=7)
    for c, m, l in zip(cvars, means, labels):
        ax.annotate(l, (c, m), textcoords="offset points", xytext=(8, 4),
                    fontsize=8)
    ax.set_xlabel("CVaR @ 5% (worst-tail revenue $)")
    ax.set_ylabel("expected revenue ($)")
    return ax


def drl_training_curve(reward_history, baseline_mean, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5))
    h = np.array(reward_history)
    # rolling mean of last 100 episodes
    if len(h) >= 50:
        rolling = np.convolve(h, np.ones(50) / 50, mode="valid")
        ax.plot(np.arange(len(rolling)) + 49, rolling, color=ACCENT, lw=1.4,
                label="50-episode rolling mean")
    ax.scatter(np.arange(len(h)), h, color=ACCENT, alpha=0.18, s=6)
    ax.axhline(baseline_mean, color=WARN, ls="--", lw=1.0,
               label=f"greedy baseline mean: {baseline_mean:,.0f}")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return ($)")
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
