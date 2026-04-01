# MSE_apply

A research pipeline that extracts **RCMSE + statistical/spectral features** from short EEG time series (≈68 points) and classifies artifact types using subject-wise cross-validation.

---

## Data

| Path                                   | Contents                                                                                |
| -------------------------------------- | --------------------------------------------------------------------------------------- |
| `data/data_records.csv`                | Artifact segment timestamps per subject/session (23 subjects × 4 sessions = 92 records) |
| `data/eye_artifact_data_original_mff/` | Raw EEG recordings (.mff, 1000 Hz, 129 ch)                                              |
| `data/SUB*_hrf.mat`                    | HRF fMRI data (26 subjects, secondary)                                                  |
| `data/BOLD/`, `data/ASL/`              | Resting-state fMRI NIfTI (secondary)                                                    |

**Artifact labels** (4-class): `0=blink`, `1=rolling`, `2=muscle`, `3=mix`

---

## File Structure

```
MSE_apply/
├── prd.md                          # project requirements document
├── README.md                       # this file
│
├── 01_data_loader.py               # load EEG .mff → resample to 68 points
├── 02_feature_extraction.py        # RCMSE + statistical + spectral features
├── 03_augmentation.py              # weak data augmentation (jitter/scale/shift)
├── 04_classification.py            # ML training + subject-wise CV + ablation
├── 05_evaluation.py                # visualization and report generation
└── run_all.py                      # full pipeline entry point
```

---

## Installation

```bash
pip install numpy scipy pandas scikit-learn matplotlib mne defusedxml xgboost
```

---

## Usage

```bash
# Full run (23 subjects, LOSO CV)
python run_all.py

# GroupKFold + data augmentation
python run_all.py --cv group_kfold --k 5 --augment --n_aug 10

# Quick test (first 2 subjects only)
python run_all.py --fast --cv group_kfold --k 2
```

### Options

| Option        | Default | Description                                     |
| ------------- | ------- | ----------------------------------------------- |
| `--cv`        | `loso`  | `loso` (Leave-One-Subject-Out) or `group_kfold` |
| `--k`         | `5`     | Number of folds for GroupKFold                  |
| `--max_scale` | `3`     | Maximum RCMSE scale (1–3 recommended for 68-pt) |
| `--augment`   | off     | Enable data augmentation                        |
| `--n_aug`     | `10`    | Number of augmented samples per subject         |
| `--fast`      | off     | Load first 2 subjects only (quick dev test)     |

---

## Output

Saved to `results/` after execution:

| File                     | Contents                                         |
| ------------------------ | ------------------------------------------------ |
| `ablation_summary.csv`   | Mean ± std Accuracy / F1 per condition and model |
| `ablation_plot.png`      | Feature ablation bar plot                        |
| `feature_importance.png` | RF top-20 feature importance bar chart           |

---

## Results (fast mode: 2 subjects, GroupKFold k=2)

> `python run_all.py --fast --cv group_kfold --k 2 --max_scale 3`

### Feature Ablation (Accuracy mean ± std)

| Condition         | LR                | SVM           | RF            | XGB           |
| ----------------- | ----------------- | ------------- | ------------- | ------------- |
| entropy_only      | 0.188 ± 0.000     | 0.156 ± 0.031 | 0.250 ± 0.063 | 0.250 ± 0.063 |
| stats_only        | 0.156 ± 0.094     | 0.188 ± 0.063 | 0.188 ± 0.063 | 0.188 ± 0.000 |
| **spectral_only** | **0.438 ± 0.063** | 0.344 ± 0.094 | 0.406 ± 0.031 | 0.313 ± 0.063 |
| ent+stats         | 0.188 ± 0.063     | 0.250 ± 0.000 | 0.156 ± 0.031 | 0.219 ± 0.031 |
| **ent+spec**      | **0.406 ± 0.031** | 0.219 ± 0.031 | 0.406 ± 0.094 | 0.313 ± 0.000 |
| all_features      | 0.313 ± 0.063     | 0.188 ± 0.063 | 0.219 ± 0.031 | 0.219 ± 0.094 |

> 4-class problem (chance level = 0.25). Spectral features dominate; the ent+spec combination consistently outperforms entropy alone.

### Feature Ablation Plot

<img width="2100" height="750" alt="image" src="https://github.com/user-attachments/assets/e14184d1-912e-424f-aef3-6944c1985f34" />


### Feature Importance (Random Forest, all data)

<img width="1500" height="900" alt="image" src="https://github.com/user-attachments/assets/0a3cdf46-c814-4f6b-89c1-1f7a88510859" />


> Beta/alpha band power and spectral entropy rank highest — RCMSE (scale 1–3) also appears within the top 10.

---

## Feature Extraction Structure

One time series (68 points) → **20-dimensional** feature vector

| Group    | Features                                                               | Dim |
| -------- | ---------------------------------------------------------------------- | --- |
| RCMSE    | scale 1, 2, 3                                                          | 3   |
| Stats    | mean, std, skew, kurt, p25, p75, iqr, min, max, range                  | 10  |
| Spectral | delta/theta/alpha/beta power, total power, spectral entropy, peak freq | 7   |

> `channel_avg=True` (default): average 129 channels → 1D → 20D  
> `channel_avg=False`: extract per channel → 20 × 129 = 2580D

---

## Classification Models

- Logistic Regression (L2, C=1)
- SVM (RBF kernel)
- Random Forest (100 estimators)
- XGBoost (100 rounds, depth 3)

Validation: subject-wise CV (`groups` = subject ID) — prevents data leakage across subjects
