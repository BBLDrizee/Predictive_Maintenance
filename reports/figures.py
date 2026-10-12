"""Charts for the progress report, built from results/ and the exported models.

Static PDF charts: thin bars, recessive grid, units on every axis, colours from the
validated reference palette in fixed categorical order (one hue per model family).
Red/amber/green appear only for risk tiers.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from turborul.reporting import BUCKET_LABEL, MODEL_LABEL, MODEL_ORDER  # noqa: E402,F401

DEEP_MODELS = ("lstm", "cnn")
COLOR = {"linear": "#2a78d6", "xgb": "#eb6834", "lstm": "#1baf7a", "cnn": "#eda100"}  # categorical slots 1-4
TIER_COLOR = {"Critical": "#d03b3b", "Watch": "#fab219", "Healthy": "#0ca30c"}
UP, DOWN = "#2a78d6", "#e87ba4"  # contribution pushes RUL up / down
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.5,
})


def _png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def rmse_by_subset_chart(finals: pd.DataFrame, target: float = 15) -> bytes:
    """Test RMSE of each model's final configuration, grouped by subset (mean ± std over seeds)."""
    subsets = sorted(finals.subset.unique())
    models = [m for m in MODEL_ORDER if m in set(finals.model)]
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    width = 0.8 / len(models)
    x = np.arange(len(subsets))
    for i, m in enumerate(models):
        d = finals[finals.model == m].set_index("subset").reindex(subsets)
        xs = x + (i - (len(models) - 1) / 2) * width
        ax.bar(xs, d.test_rmse, width * 0.9, color=COLOR[m], label=MODEL_LABEL[m], linewidth=0)
        ax.errorbar(xs, d.test_rmse, yerr=d.test_rmse_std.fillna(0), fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
    ax.axhline(target, color=INK_MUTED, lw=0.8, ls="--", label=f"FD001 target ≤ {target:g}")
    ax.set_xticks(x, subsets)
    ax.set_ylabel("Test RMSE (cycles)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(models) + 1, loc="upper left", bbox_to_anchor=(0, 1.13))
    return _png(fig)


def scatter_grid(preds: dict[str, pd.DataFrame], labels: dict[str, str], critical: int) -> bytes:
    """Predicted vs true RUL with 90% ranges, one panel per subset (deployed model, seed 0)."""
    fig, axes = plt.subplots(1, len(preds), figsize=(7.0, 2.3), sharey=True)
    lim = max(max(d.true_rul.max(), d.high.max()) for d in preds.values()) + 8
    for ax, (subset, d) in zip(np.atleast_1d(axes), preds.items()):
        d = d[d.seed == d.seed.min()]
        model = labels[subset]
        ax.axhspan(0, critical, color=GRID, alpha=0.5, lw=0)
        ax.plot([0, lim], [0, lim], color=INK_MUTED, lw=0.8, ls="--")
        ax.errorbar(d.true_rul, d.pred_rul, yerr=[d.pred_rul - d.low, d.high - d.pred_rul], fmt="none",
                    ecolor="#c9c8c1", elinewidth=0.5)
        ax.scatter(d.true_rul, d.pred_rul, s=7, color=COLOR[model], edgecolor="white", linewidth=0.3, zorder=3)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="True RUL (cycles)")
        ax.set_title(f"{subset} · {MODEL_LABEL[model]}", fontsize=8)
        ax.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel("Predicted RUL (cycles)")
    fig.tight_layout()
    return _png(fig)


def bucket_grid(buckets: pd.DataFrame) -> bytes:
    """RMSE by true-RUL bucket per subset (final configurations, mean over seeds)."""
    subsets = sorted(buckets.subset.unique())
    models = [m for m in MODEL_ORDER if m in set(buckets.model)]
    fig, axes = plt.subplots(1, len(subsets), figsize=(7.0, 2.3), sharey=True)
    width = 0.8 / len(models)
    for ax, s in zip(np.atleast_1d(axes), subsets):
        b = buckets[buckets.subset == s]
        x = np.arange(len(BUCKET_LABEL))
        for i, m in enumerate(models):
            d = b[b.model == m].set_index("bucket").reindex(BUCKET_LABEL)
            ax.bar(x + (i - (len(models) - 1) / 2) * width, d.rmse, width * 0.92, color=COLOR[m], label=MODEL_LABEL[m], linewidth=0)
        ax.set_xticks(x, BUCKET_LABEL, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(s, fontsize=8)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("RMSE (cycles)")
    fig.supxlabel("True RUL bucket (cycles)", fontsize=8, color=INK)
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=7)
    fig.tight_layout()
    return _png(fig)


def coverage_chart(finals: pd.DataFrame, target: float = 0.9) -> bytes:
    """Test coverage of the 90% ranges, against uncapped and capped truth."""
    subsets = sorted(finals.subset.unique())
    models = [m for m in MODEL_ORDER if m in set(finals.model)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    for ax, col, title in [(axes[0], "coverage", "vs true RUL (uncapped)"), (axes[1], "coverage_capped", "vs true RUL capped at the model's cap")]:
        for i, m in enumerate(models):
            d = finals[finals.model == m].set_index("subset").reindex(subsets)
            ax.plot(np.arange(len(subsets)) + (i - 1.5) * 0.09, d[col], "o", color=COLOR[m], ms=5, label=MODEL_LABEL[m])
        ax.axhline(target, color=INK_MUTED, lw=0.8, ls="--")
        ax.text(len(subsets) - 0.6, target + 0.01, "target 90%", fontsize=6.5, color=INK_MUTED, ha="right")
        ax.set_xticks(range(len(subsets)), subsets)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Test engines inside range")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=7)
    fig.tight_layout()
    return _png(fig)


def importance_grid(importance: dict[str, pd.DataFrame], labels: dict[str, str], top: int = 6) -> bytes:
    """Mean |contribution| per sensor across each subset's fleet (deployed model)."""
    fig, axes = plt.subplots(1, len(importance), figsize=(7.0, 2.0))
    for ax, (subset, imp) in zip(np.atleast_1d(axes), importance.items()):
        d = imp.head(top).iloc[::-1]
        ax.barh([f"{r.sensor} {r.symbol}" for _, r in d.iterrows()], d.mean_abs_contribution, color=COLOR[labels[subset]], height=0.6)
        ax.set_title(f"{subset} · {MODEL_LABEL[labels[subset]]}", fontsize=8)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("Mean |contribution| to predicted RUL across the fleet (cycles)", fontsize=8, color=INK)
    fig.tight_layout()
    return _png(fig)


def explanation_chart(top_sensors: list[dict], title: str, base: float, rul: float) -> bytes:
    """One engine's top sensor contributions, signed."""
    d = pd.DataFrame(top_sensors).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.9, 2.3))
    ax.barh([f"{r.sensor} · {r.symbol}" for _, r in d.iterrows()], d.contribution,
            color=[UP if v > 0 else DOWN for v in d.contribution], height=0.6)
    span = d.contribution.abs().max()
    for y, v in enumerate(d.contribution):
        ax.text(v + (0.03 if v > 0 else -0.03) * span, y, f"{v:+.1f}", va="center", ha="left" if v > 0 else "right",
                fontsize=6.5, color=INK)
    ax.axvline(0, color=INK_MUTED, lw=0.6)
    ax.set_xlim(-1.35 * span, 1.35 * span) if (d.contribution > 0).any() else ax.set_xlim(-1.35 * span, 0.25 * span)
    ax.set_xlabel("Contribution to predicted RUL (cycles)")
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", visible=False)
    return _png(fig)


def trajectory_chart(traj: pd.DataFrame, title: str, critical: int, watch: int) -> bytes:
    """Predicted RUL and 90% range over one engine's life, with the true RUL."""
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.fill_between(traj.cycle, traj.low, traj.high, color=UP, alpha=0.15, lw=0, label="90% range")
    ax.plot(traj.cycle, traj.rul, color=UP, lw=1.5, label="Predicted")
    ax.plot(traj.cycle, traj.true_rul, color=INK_MUTED, lw=1.2, ls="--", label="True")
    for tier, y in [("Critical", critical), ("Watch", watch)]:
        ax.axhline(y, color=TIER_COLOR[tier], lw=0.8, ls=":")
    ax.set(xlabel="Cycle", ylabel="RUL (cycles)")
    ax.set_title(title, fontsize=8)
    ax.legend(loc="upper right", fontsize=6.5)
    return _png(fig)
