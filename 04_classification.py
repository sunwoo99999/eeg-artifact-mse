"""
04_classification.py
Subject-wise CV (LOSO / GroupKFold) + comparison of 4 ML models.
Feature ablation: Entropy only vs. Entropy+Stats+Spectral
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    LeaveOneGroupOut,
    GroupKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import importlib.util, pathlib, sys
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — skipping XGBoost.")

# Load 02_feature_extraction.py (numbered filename not directly importable)
_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("_feat_ext", _here / "02_feature_extraction.py")
_feat_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feat_mod)
extract_all = _feat_mod.extract_all

# ────────────────────────────────────────────────────────────────────────────
# Model definitions
# ────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    models = {
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
    }
    if HAS_XGB:
        models["XGB"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=100, max_depth=3, random_state=42,
                eval_metric="logloss", verbosity=0,
            )),
        ])
    return models


# ────────────────────────────────────────────────────────────────────────────
# CV utilities
# ────────────────────────────────────────────────────────────────────────────

def subject_wise_cv(
    X_feat: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: dict,
    cv_strategy: str = "loso",
    n_splits: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run subject-wise cross-validation.

    Parameters
    ----------
    X_feat     : (N, D)
    y          : (N,)
    groups     : (N,)  subject group IDs
    models     : dict of name -> sklearn Pipeline
    cv_strategy: "loso" (Leave-One-Subject-Out) or "group_kfold"
    n_splits   : k for GroupKFold (ignored for loso)

    Returns
    -------
    results_df : DataFrame (model × metric)
    """
    if cv_strategy == "loso":
        cv = LeaveOneGroupOut()
    else:
        cv = GroupKFold(n_splits=n_splits)

    records = []
    n_classes = len(np.unique(y))

    for name, pipe in models.items():
        accs, f1s, aucs = [], [], []
        for train_idx, test_idx in cv.split(X_feat, y, groups=groups):
            X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)

            accs.append(accuracy_score(y_te, y_pred))
            f1s.append(f1_score(y_te, y_pred, average="weighted", zero_division=0))

            if n_classes == 2 and hasattr(pipe, "predict_proba"):
                try:
                    y_prob = pipe.predict_proba(X_te)[:, 1]
                    aucs.append(roc_auc_score(y_te, y_prob))
                except Exception:
                    pass

        rec = {
            "model": name,
            "acc_mean": np.mean(accs),
            "acc_std":  np.std(accs),
            "f1_mean":  np.mean(f1s),
            "f1_std":   np.std(f1s),
        }
        if aucs:
            rec["auc_mean"] = np.mean(aucs)
            rec["auc_std"]  = np.std(aucs)
        records.append(rec)

        if verbose:
            auc_str = f"  AUC {rec.get('auc_mean', np.nan):.3f}±{rec.get('auc_std', 0):.3f}" if aucs else ""
            print(f"  {name:6s}: Acc {rec['acc_mean']:.3f}±{rec['acc_std']:.3f}"
                  f"  F1 {rec['f1_mean']:.3f}±{rec['f1_std']:.3f}{auc_str}")

    return pd.DataFrame(records).set_index("model")


# ────────────────────────────────────────────────────────────────────────────
# Feature ablation
# ────────────────────────────────────────────────────────────────────────────

ABLATION_CONDITIONS = {
    "entropy_only":    dict(include_entropy=True,  include_stats=False, include_spectral=False),
    "stats_only":      dict(include_entropy=False, include_stats=True,  include_spectral=False),
    "spectral_only":   dict(include_entropy=False, include_stats=False, include_spectral=True),
    "ent+stats":       dict(include_entropy=True,  include_stats=True,  include_spectral=False),
    "ent+spec":        dict(include_entropy=True,  include_stats=False, include_spectral=True),
    "all_features":    dict(include_entropy=True,  include_stats=True,  include_spectral=True),
}


def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_scale: int = 3,
    fs: float = 1.0,
    cv_strategy: str = "loso",
    n_splits: int = 5,
) -> dict:
    """
    Run CV for each feature combination and return results.

    Returns
    -------
    ablation_results : dict {condition_name -> DataFrame}
    """
    ablation_results = {}
    models = get_models()

    for cond_name, feat_kwargs in ABLATION_CONDITIONS.items():
        print(f"\n{'='*55}")
        print(f"  Condition: {cond_name}")
        print(f"{'='*55}")

        X_feat = extract_all(X, max_scale=max_scale, fs=fs, **feat_kwargs)
        print(f"  Feature dim: {X_feat.shape[1]}")

        # Replace NaN with column mean
        col_means = np.nanmean(X_feat, axis=0)
        nan_mask = np.isnan(X_feat)
        X_feat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        df_cv = subject_wise_cv(
            X_feat, y, groups, models,
            cv_strategy=cv_strategy, n_splits=n_splits
        )
        ablation_results[cond_name] = df_cv

    return ablation_results


def summarize_ablation(ablation_results: dict) -> pd.DataFrame:
    """
    Consolidate ablation results into a single summary DataFrame.
    """
    rows = []
    for cond, df in ablation_results.items():
        for model_name, row in df.iterrows():
            rows.append({
                "condition": cond,
                "model": model_name,
                "acc_mean": row["acc_mean"],
                "acc_std":  row["acc_std"],
                "f1_mean":  row["f1_mean"],
                "f1_std":   row["f1_std"],
            })
    summary = pd.DataFrame(rows)
    return summary


if __name__ == "__main__":
    # Minimal smoke test with random data
    rng = np.random.default_rng(42)
    N, T, R = 26, 7, 13
    X_dummy = rng.standard_normal((N, T, R))
    y_dummy = np.array([0] * 13 + [1] * 13)
    groups  = np.arange(N)

    print("Running ablation (smoke test)...")
    results = run_ablation(X_dummy, y_dummy, groups, cv_strategy="loso")
    summary = summarize_ablation(results)
    print("\n=== Summary (Acc mean) ===")
    pivot = summary.pivot(index="condition", columns="model", values="acc_mean")
    print(pivot.to_string())
