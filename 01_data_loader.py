"""
01_data_loader.py
EEG .mff file loader and artifact segment extraction.

data_records.csv timestamp format: "[min,sec;min,sec]"
  e.g. "[0,45;1,50]" → 0 min 45 sec to 1 min 50 sec

Output shape: X (N, 68, C)  N=number of samples, C=number of channels (or 1 if channel-avg)
              y (N,)        0=blink, 1=rolling, 2=muscle, 3=mix
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import scipy.signal as sig

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MFF_DIR  = os.path.join(DATA_DIR, "eye_artifact_data_original_mff")
CSV_PATH = os.path.join(DATA_DIR, "data_records.csv")

# ── Fixed output length ──────────────────────────────────────────────────────
TARGET_LEN = 68          # time points after resampling
ARTIFACT_COLS = {
    "s1blink":   0,
    "s2rolling": 1,
    "s3muscle":  2,
    "s4mix":     3,
}
LABEL_NAMES = {0: "blink", 1: "rolling", 2: "muscle", 3: "mix"}


def _parse_timestamp(ts_str: str) -> tuple[float, float]:
    """
    Parse a "[min,sec;min,sec]" format string into seconds (start, end).
    e.g. "[0,45;1,50]" → (45.0, 110.0)
    """
    ts_str = ts_str.strip().strip("[]")
    parts = ts_str.split(";")
    results = []
    for p in parts:
        m, s = p.strip().split(",")
        results.append(int(m) * 60 + int(s))
    return float(results[0]), float(results[1])


def _find_mff_path(subject_id: int, session: int, mff_dir: str) -> str | None:
    """
    Search for the .mff folder path corresponding to subject_id and session.
    Filename pattern: ar{sub}_s{session}_1000_*.mff
    """
    pattern = re.compile(
        rf"ar{subject_id}_s{session}_1000_.*\.mff$", re.IGNORECASE
    )
    for entry in os.listdir(mff_dir):
        if pattern.match(entry):
            return os.path.join(mff_dir, entry)
    return None


def _load_segment(
    mff_path: str,
    t_start: float,
    t_end: float,
    channel_avg: bool = True,
    target_len: int = TARGET_LEN,
) -> np.ndarray | None:
    """
    Load the [t_start, t_end] second segment from a .mff file
    and resample to target_len points.

    Parameters
    ----------
    channel_avg : True  → average all channels → (target_len,)
                  False → keep all channels   → (target_len, n_channels)

    Returns  None if loading fails.
    """
    try:
        import mne
        raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]

        # Sample indices
        i_start = int(t_start * sfreq)
        i_end   = min(int(t_end   * sfreq), raw.n_times)
        if i_end <= i_start:
            return None

        data = raw.get_data()[:, i_start:i_end]   # (n_ch, n_samples)

        if channel_avg:
            data = data.mean(axis=0)               # (n_samples,)
        else:
            data = data.T                          # (n_samples, n_ch)

        # Resample to target_len
        if data.ndim == 1:
            resampled = sig.resample(data, target_len)
        else:
            resampled = sig.resample(data, target_len, axis=0)

        return resampled

    except Exception as e:
        print(f"  [WARN] Failed to load {os.path.basename(mff_path)}: {e}")
        return None


def load_eeg_dataset(
    csv_path:    str  = CSV_PATH,
    mff_dir:     str  = MFF_DIR,
    channel_avg: bool = True,
    target_len:  int  = TARGET_LEN,
    verbose:     bool = True,
) -> tuple:
    """
    Load the EEG artifact dataset.

    Returns
    -------
    X      : np.ndarray (N, T)            channel_avg=True
             np.ndarray (N, T, C)         channel_avg=False
    y      : np.ndarray (N,)              0=blink … 3=mix
    groups : np.ndarray (N,)              subject ID (for CV grouping)
    meta   : list of dict                 {subject, session, artifact_type, t_start, t_end}
    """
    df = pd.read_csv(csv_path)
    X_list, y_list, groups_list, meta_list = [], [], [], []

    total = len(df) * len(ARTIFACT_COLS)
    done  = 0

    for _, row in df.iterrows():
        sub     = int(row["subject"])
        session = int(row["session"])
        mff_path = _find_mff_path(sub, session, mff_dir)

        if mff_path is None:
            if verbose:
                print(f"  [WARN] No .mff for sub={sub}, session={session}")
            done += len(ARTIFACT_COLS)
            continue

        for col_name, label in ARTIFACT_COLS.items():
            done += 1
            ts_str = row[col_name]
            if pd.isna(ts_str):
                continue

            try:
                t_start, t_end = _parse_timestamp(str(ts_str))
            except Exception:
                continue

            segment = _load_segment(
                mff_path, t_start, t_end,
                channel_avg=channel_avg,
                target_len=target_len,
            )

            if segment is None:
                continue

            X_list.append(segment)
            y_list.append(label)
            groups_list.append(sub)
            meta_list.append({
                "subject":       sub,
                "session":       session,
                "artifact_type": LABEL_NAMES[label],
                "t_start":       t_start,
                "t_end":         t_end,
            })

        if verbose:
            pct = 100 * done / total
            print(f"  Progress: {done}/{total} ({pct:.0f}%)  sub={sub} session={session}", end="\r")

    print()
    if not X_list:
        raise RuntimeError("No segments loaded. Check .mff paths and CSV timestamps.")

    X      = np.array(X_list)
    y      = np.array(y_list)
    groups = np.array(groups_list)

    if verbose:
        print(f"[INFO] Loaded {len(X)} segments  |  X.shape={X.shape}")
        print(f"[INFO] Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, groups, meta_list


if __name__ == "__main__":
    print("Loading EEG dataset (first subject only for quick test)...")
    df = pd.read_csv(CSV_PATH)
    first_row = df.iloc[:1]
    first_row.to_csv("/tmp/_test_records.csv", index=False)

    X, y, groups, meta = load_eeg_dataset(
        csv_path=CSV_PATH,
        verbose=True,
    )
    print("X shape:", X.shape)
    print("y:", y[:10])
    print("Groups:", groups[:10])

