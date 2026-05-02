from __future__ import annotations
import numpy as np


def _ensure_not_all_missing(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    row_all = mask.all(axis=1)
    for r in np.where(row_all)[0]:
        j = int(rng.integers(0, mask.shape[1]))
        mask[r, j] = False
    return mask


def apply_missingness(X: np.ndarray, mechanism: str, missing_rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = X.copy().astype(float)
    mechanism = mechanism.upper()
    if mechanism == "MCAR":
        mask = rng.random(X.shape) < missing_rate
    elif mechanism == "MAR":
        z = np.nan_to_num(X, nan=np.nanmean(X, axis=0, keepdims=True))
        z = (z - np.nanmean(z, axis=0, keepdims=True)) / (np.nanstd(z, axis=0, keepdims=True) + 1e-6)
        probs = 1.0 / (1.0 + np.exp(-z))
        probs = np.clip(0.05 + probs * missing_rate, 0.0, 0.95)
        mask = rng.random(X.shape) < probs
    elif mechanism == "MNAR":
        z = np.nan_to_num(X, nan=np.nanmean(X, axis=0, keepdims=True))
        centered = z - np.nanmean(z, axis=0, keepdims=True)
        probs = np.abs(centered) / (np.nanstd(z, axis=0, keepdims=True) + 1e-6)
        probs = probs / (np.nanmax(probs, axis=0, keepdims=True) + 1e-6)
        probs = np.clip(0.05 + probs * missing_rate, 0.0, 0.95)
        mask = rng.random(X.shape) < probs
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    mask = _ensure_not_all_missing(mask, rng)
    X[mask] = np.nan
    return X, mask.astype(int)
