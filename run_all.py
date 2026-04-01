"""
run_all.py
Entry point for the full MSE EEG artifact classification pipeline.

Usage:
    python run_all.py [--cv loso|group_kfold] [--augment] [--max_scale 3]
    python run_all.py --fast   # load first 2 subjects only (quick test)
"""

import argparse
import importlib.util
import pathlib
import numpy as np

HERE = pathlib.Path(__file__).parent


def _load(fname, alias):
    spec = importlib.util.spec_from_file_location(alias, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="MSE EEG artifact classification pipeline")
    parser.add_argument("--cv",        default="loso", choices=["loso", "group_kfold"])
    parser.add_argument("--k",         default=5, type=int)
    parser.add_argument("--augment",   action="store_true")
    parser.add_argument("--n_aug",     default=10, type=int)
    parser.add_argument("--max_scale", default=3, type=int)
    parser.add_argument("--fast",      action="store_true",
                        help="Quick test: load only first 4 records (1 subject)")
    args = parser.parse_args()

    # ── Load modules ──────────────────────────────────────────────────────
    loader_mod = _load("01_data_loader.py",       "_loader")
    aug_mod    = _load("03_augmentation.py",       "_aug")
    eval_mod   = _load("05_evaluation.py",         "_eval")
    cls_mod    = _load("04_classification.py",     "_cls")
    feat_mod   = _load("02_feature_extraction.py", "_feat")

    # ── Step 1: Load EEG data ─────────────────────────────────────────────
    print("\n[STEP 1] Loading EEG artifact segments...")
    import pandas as pd
    csv_path = str(HERE / "data" / "data_records.csv")
    mff_dir  = str(HERE / "data" / "eye_artifact_data_original_mff")

    if args.fast:
        # Limit to first 2 subjects (8 sessions) for fast test
        df_full = pd.read_csv(csv_path)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df_full.iloc[:8].to_csv(tmp.name, index=False)
        tmp.close()
        csv_path = tmp.name
        print("  [FAST MODE] Using first 2 subjects only.")

    X, y, groups, meta = loader_mod.load_eeg_dataset(
        csv_path=csv_path, mff_dir=mff_dir, channel_avg=True, verbose=True,
    )
    # X: (N, 68)  →  reshape to (N, 68, 1) for compatibility with extract_all
    X = X[:, :, np.newaxis]   # (N, T, 1)
    print(f"  X: {X.shape}  |  y unique: {np.unique(y)}")

    # ── Step 2: Augmentation ──────────────────────────────────────────────
    if args.augment:
        print(f"\n[STEP 2] Augmenting (n_aug={args.n_aug})...")
        X, y = aug_mod.augment_dataset(X, y, n_aug=args.n_aug, random_state=42)
        groups = np.repeat(groups, args.n_aug + 1)
    else:
        print("\n[STEP 2] Skipping augmentation (--augment to enable)")

    # ── Step 3: Feature ablation ───────────────────────────────────────────
    print(f"\n[STEP 3] Running feature ablation (max_scale={args.max_scale}, cv={args.cv})...")
    ablation_results = cls_mod.run_ablation(
        X, y, groups,
        max_scale=args.max_scale,
        fs=1.0,
        cv_strategy=args.cv,
        n_splits=args.k,
    )

    # ── Step 4: Reports ────────────────────────────────────────────────────
    print("\n[STEP 4] Generating reports...")
    summary = cls_mod.summarize_ablation(ablation_results)
    eval_mod.print_summary_table(summary)
    eval_mod.plot_ablation(summary)
    eval_mod.save_csv(summary)

    print("\n[STEP 4b] Feature importance...")
    X_all = feat_mod.extract_all(X, max_scale=args.max_scale, fs=1.0,
                                  include_entropy=True, include_stats=True, include_spectral=True)
    col_means = np.nanmean(X_all, axis=0)
    X_all[np.isnan(X_all)] = np.take(col_means, np.where(np.isnan(X_all))[1])
    feat_names = feat_mod.build_feature_names(max_scale=args.max_scale, n_rois=X.shape[2])
    eval_mod.plot_feature_importance(X_all, y, feat_names)

    print("\n[DONE] Pipeline complete. Results in ./results/")


if __name__ == "__main__":
    main()
