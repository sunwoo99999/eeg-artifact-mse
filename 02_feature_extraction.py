"""
02_feature_extraction.py
RCMSE + statistical + spectral feature extraction.

Input time series: 1D array of shape (T,) (T ≈ 68 for downsampled EEG)
"""

import numpy as np
from scipy import stats, signal
from itertools import product


# ────────────────────────────────────────────────────────────────────────────
# RCMSE (Refined Composite Multiscale Sample Entropy)
# ────────────────────────────────────────────────────────────────────────────

def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Sample Entropy of 1D time series x.
    r = r_factor * std(x)
    Returns NaN if computation is undefined (too few matches).
    """
    n = len(x)
    if n < m + 2:
        return np.nan

    r = r_factor * np.std(x, ddof=0)
    if r == 0:
        return np.nan

    def _count_matches(length):
        count = 0
        for i in range(n - length):
            template = x[i:i + length]
            for j in range(i + 1, n - length + 1):
                if np.max(np.abs(x[j:j + length] - template)) < r:
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)

    if B == 0:
        return np.nan
    return -np.log(A / B) if A > 0 else np.nan


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    """
    Standard coarse-graining: non-overlapping mean windows of size `scale`.
    """
    n = (len(x) // scale) * scale
    return x[:n].reshape(-1, scale).mean(axis=1)


def rcmse(x: np.ndarray, max_scale: int = 3, m: int = 2, r_factor: float = 0.2) -> np.ndarray:
    """
    Refined Composite Multiscale Sample Entropy (RCMSE).

    For short series (len < 20), max_scale is automatically capped.
    Returns array of length max_scale (NaN where undefined).
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return np.full(max_scale, np.nan)

    # Cap max_scale to avoid degenerate coarse-graining
    safe_max = max(1, len(x) // 4)
    actual_max = min(max_scale, safe_max)

    entropies = []
    for s in range(1, actual_max + 1):
        # RCMSE: average SE over all `s` starting-point shifts
        se_vals = []
        for k in range(s):
            cg = _coarse_grain(x[k:], s)
            if len(cg) >= m + 2:
                se = _sample_entropy(cg, m=m, r_factor=r_factor)
                se_vals.append(se)
        if se_vals:
            valid = [v for v in se_vals if not np.isnan(v)]
            entropies.append(np.mean(valid) if valid else np.nan)
        else:
            entropies.append(np.nan)

    # Pad with NaN if actual_max < max_scale
    while len(entropies) < max_scale:
        entropies.append(np.nan)

    return np.array(entropies)


# ────────────────────────────────────────────────────────────────────────────
# Statistical features
# ────────────────────────────────────────────────────────────────────────────

def statistical_features(x: np.ndarray) -> np.ndarray:
    """
    Returns: [mean, std, skewness, kurtosis, p25, p75, iqr, min, max, range]
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.zeros(10)

    p25, p75 = np.percentile(x, [25, 75])
    feats = np.array([
        np.mean(x),
        np.std(x, ddof=1) if len(x) > 1 else 0.0,
        float(stats.skew(x)),
        float(stats.kurtosis(x)),
        p25,
        p75,
        p75 - p25,
        np.min(x),
        np.max(x),
        np.ptp(x),          # peak-to-peak range
    ])
    return feats


# ────────────────────────────────────────────────────────────────────────────
# Spectral features
# ────────────────────────────────────────────────────────────────────────────

def spectral_features(x: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Spectral features via Welch PSD (or periodogram for very short series).
    Returns: [delta_power, theta_power, alpha_power, beta_power,
              total_power, spectral_entropy, peak_freq]

    Band definitions (Hz) — adapted for fs=1 Hz (HRF-like) or higher:
      delta : 0 – 0.1*nyq
      theta : 0.1 – 0.2*nyq
      alpha : 0.2 – 0.4*nyq
      beta  : 0.4 – nyq
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    nyq = fs / 2.0

    if n < 4:
        return np.zeros(7)

    # Use periodogram for short series (Welch needs nperseg >= 4)
    nperseg = min(n, max(4, n // 2))
    try:
        freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    except Exception:
        freqs, psd = signal.periodogram(x, fs=fs)

    _integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return _integrate(psd[mask], freqs[mask]) if mask.any() else 0.0

    bands = {
        "delta": (0.0,       0.1 * nyq),
        "theta": (0.1 * nyq, 0.2 * nyq),
        "alpha": (0.2 * nyq, 0.4 * nyq),
        "beta":  (0.4 * nyq, nyq),
    }
    powers = {k: band_power(*v) for k, v in bands.items()}
    total = sum(powers.values()) + 1e-12

    # Spectral entropy (normalized PSD)
    psd_norm = psd / (psd.sum() + 1e-12)
    sp_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))

    peak_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0

    return np.array([
        powers["delta"],
        powers["theta"],
        powers["alpha"],
        powers["beta"],
        total,
        sp_entropy,
        peak_freq,
    ])


# Feature names for logging
STAT_NAMES  = ["mean", "std", "skew", "kurt", "p25", "p75", "iqr", "min", "max", "range"]
SPEC_NAMES  = ["delta_pw", "theta_pw", "alpha_pw", "beta_pw", "total_pw", "sp_ent", "peak_f"]


def extract_subject_features(
    x_roi: np.ndarray,
    max_scale: int = 3,
    fs: float = 1.0,
    include_entropy: bool = True,
    include_stats: bool = True,
    include_spectral: bool = True,
) -> np.ndarray:
    """
    Extract all features from a single time series (T,).

    Returns
    -------
    feature vector : np.ndarray (D,)
    """
    parts = []
    if include_entropy:
        parts.append(rcmse(x_roi, max_scale=max_scale))
    if include_stats:
        parts.append(statistical_features(x_roi))
    if include_spectral:
        parts.append(spectral_features(x_roi, fs=fs))

    return np.concatenate(parts) if parts else np.array([])


def build_feature_names(max_scale: int = 3, n_rois: int = 13) -> list[str]:
    """
    Return column names for the feature matrix.
    The same feature set is repeated for each ROI/channel.
    """
    per_roi = (
        [f"rcmse_s{s+1}" for s in range(max_scale)]
        + STAT_NAMES
        + SPEC_NAMES
    )
    return [f"roi{r:02d}_{f}" for r in range(n_rois) for f in per_roi]


def extract_all(
    X: np.ndarray,
    max_scale: int = 3,
    fs: float = 1.0,
    include_entropy: bool = True,
    include_stats: bool = True,
    include_spectral: bool = True,
) -> np.ndarray:
    """
    X : (N, T, R)  N=subjects, T=time points, R=number of ROIs/channels
    Returns : (N, D)  D = R × (MSE_dim + stats_dim + spec_dim)
    """
    N, T, R = X.shape
    rows = []
    for i in range(N):
        feats_per_subject = []
        for r in range(R):
            ts = X[i, :, r]
            f = extract_subject_features(
                ts,
                max_scale=max_scale,
                fs=fs,
                include_entropy=include_entropy,
                include_stats=include_stats,
                include_spectral=include_spectral,
            )
            feats_per_subject.append(f)
        rows.append(np.concatenate(feats_per_subject))
    return np.vstack(rows)


if __name__ == "__main__":
    # Quick sanity check
    rng = np.random.default_rng(42)
    ts = rng.standard_normal(7)   # 7-point HRF-like series
    print("Test time series:", ts)
    print("RCMSE (scale 1-3):", rcmse(ts, max_scale=3))
    print("Stat features:", statistical_features(ts))
    print("Spectral features:", spectral_features(ts, fs=1.0))
    print("Full feature vector len:", len(extract_subject_features(ts)))
