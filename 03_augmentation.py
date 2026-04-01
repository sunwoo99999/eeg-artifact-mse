"""
03_augmentation.py
Weak data augmentation for short time series.

Note: augmentation does not significantly increase effective MSE scale.
      Use only as a supplementary aid for stabilizing analysis.
"""

import numpy as np


def augment_jitter_scale(
    x: np.ndarray,
    n_aug: int = 20,
    noise_std: float = 0.02,
    scale_range: tuple = (0.9, 1.1),
    shift_range: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate n_aug augmented time series from 1D time series x.

    Augmentation methods:
      1. Jitter : add Gaussian noise
      2. Scaling: multiply the entire series by a random scalar
      3. Shift  : small DC offset

    Parameters
    ----------
    x          : (T,) original time series
    n_aug      : number of augmented samples to generate
    noise_std  : jitter std as a multiple of x.std()
    scale_range: (min, max) uniform scaling range
    shift_range: shift std as a multiple of x.std()
    rng        : numpy random Generator (fix seed for reproducibility)

    Returns
    -------
    augmented : (n_aug, T)  augmented series only (original excluded)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)
    x_std = np.std(x) + 1e-12
    result = []

    for _ in range(n_aug):
        aug = x.copy()
        # Jitter
        aug = aug + rng.normal(0, noise_std * x_std, size=x.shape)
        # Scaling
        scale = rng.uniform(*scale_range)
        aug = aug * scale
        # Shift
        aug = aug + rng.normal(0, shift_range * x_std)
        result.append(aug)

    return np.vstack(result)


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_aug: int = 10,
    noise_std: float = 0.02,
    scale_range: tuple = (0.9, 1.1),
    shift_range: float = 0.05,
    random_state: int = 42,
) -> tuple:
    """
    Apply augmentation to the entire dataset (N, T, R).

    Parameters
    ----------
    X : (N, T, R)
    y : (N,)

    Returns
    -------
    X_aug : (N * (1 + n_aug), T, R)  original + augmented
    y_aug : (N * (1 + n_aug),)
    """
    rng = np.random.default_rng(random_state)
    N, T, R = X.shape

    X_aug_list = [X]
    y_aug_list = [y]

    for i in range(N):
        for _ in range(n_aug):
            aug_sample = np.stack(
                [augment_jitter_scale(X[i, :, r], n_aug=1, noise_std=noise_std,
                                      scale_range=scale_range, shift_range=shift_range,
                                      rng=rng)[0]
                 for r in range(R)],
                axis=-1,
            )   # (T, R)
            X_aug_list.append(aug_sample[np.newaxis, ...])
            y_aug_list.append(np.array([y[i]]))

    X_out = np.concatenate(X_aug_list, axis=0)
    y_out = np.concatenate(y_aug_list, axis=0)
    print(f"[INFO] Augmented: {N} → {X_out.shape[0]} samples (×{n_aug + 1})")
    return X_out, y_out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    ts = rng.standard_normal(7)
    print("Original:", ts)
    aug = augment_jitter_scale(ts, n_aug=5, rng=rng)
    print("Augmented shape:", aug.shape)
    print(aug)
