"""
05_evaluation.py
Result aggregation, visualization, and report generation.
"""

import importlib.util, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load classification module ───────────────────────────────────────────────
_here = pathlib.Path(__file__).parent
def _load_mod(fname, alias):
    spec = importlib.util.spec_from_file_location(alias, _here / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cls_mod  = _load_mod("04_classification.py", "_cls")
run_ablation      = _cls_mod.run_ablation
summarize_ablation = _cls_mod.summarize_ablation


# ────────────────────────────────────────────────────────────────────────────
# Report helpers
# ────────────────────────────────────────────────────────────────────────────

def print_summary_table(summary: pd.DataFrame) -> None:
    """Print a text summary table to stdout."""
    pivot_acc = summary.pivot(index="condition", columns="model", values="acc_mean")
    pivot_f1  = summary.pivot(index="condition", columns="model", values="f1_mean")

    print("\n" + "=" * 65)
    print("  ACCURACY (mean across folds)")
    print("=" * 65)
    print(pivot_acc.round(3).to_string())
    print("\n" + "=" * 65)
    print("  F1-SCORE WEIGHTED (mean across folds)")
    print("=" * 65)
    print(pivot_f1.round(3).to_string())


def plot_ablation(summary: pd.DataFrame, output_path: str = "results/ablation_plot.png") -> None:
    """Save feature ablation comparison bar plot."""
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    conditions = summary["condition"].unique()
    models     = summary["model"].unique()
    n_cond = len(conditions)
    n_mod  = len(models)
    x = np.arange(n_cond)
    width = 0.8 / n_mod

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, label in zip(axes, ["acc_mean", "f1_mean"],
                                  ["Accuracy", "F1 (weighted)"]):
        for mi, mname in enumerate(models):
            sub = summary[summary["model"] == mname]
            sub = sub.set_index("condition").reindex(conditions)
            vals = sub[metric].values
            errs = sub[metric.replace("mean", "std")].values
            offset = (mi - n_mod / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, yerr=errs,
                   label=mname, capsize=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.set_title(f"Feature Ablation — {label}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Ablation plot saved → {output_path}")


def plot_feature_importance(
    X_feat: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    output_path: str = "results/feature_importance.png",
    top_n: int = 20,
) -> None:
    """Fit a Random Forest on all data and save top-N feature importance bar chart."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), importances[indices], color="steelblue", alpha=0.85)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [feature_names[i] if i < len(feature_names) else f"f{i}" for i in indices],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest, all data)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Feature importance plot saved → {output_path}")


def save_csv(summary: pd.DataFrame, output_path: str = "results/ablation_summary.csv") -> None:
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"[INFO] Summary CSV saved → {output_path}")


if __name__ == "__main__":
    # Quick sanity test
    rng = np.random.default_rng(42)
    N, T, R = 26, 7, 13
    X = rng.standard_normal((N, T, R))
    y = np.array([0] * 13 + [1] * 13)
    groups = np.arange(N)

    results = run_ablation(X, y, groups, cv_strategy="loso")
    summary = summarize_ablation(results)
    print_summary_table(summary)
    plot_ablation(summary)
    save_csv(summary)
