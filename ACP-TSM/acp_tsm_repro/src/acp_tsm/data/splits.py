from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split


def chronological_split(X: np.ndarray, y: np.ndarray, train_cal_cut: int | None = None, test_size: int | None = None, seed: int = 0):
    n = len(X)
    if train_cal_cut is None:
        train_cal_cut = n // 2
    if test_size is None:
        X_train_cal, y_train_cal = X[:train_cal_cut], y[:train_cal_cut]
        X_test, y_test = X[train_cal_cut:], y[train_cal_cut:]
    else:
        X_train_cal, y_train_cal = X[:train_cal_cut], y[:train_cal_cut]
        X_test, y_test = X[train_cal_cut:train_cal_cut + test_size], y[train_cal_cut:train_cal_cut + test_size]
    idx = np.arange(len(X_train_cal))
    train_idx, cal_idx = train_test_split(idx, test_size=0.5, random_state=seed, shuffle=True)
    return {
        "train": (X_train_cal[train_idx], y_train_cal[train_idx]),
        "cal": (X_train_cal[cal_idx], y_train_cal[cal_idx]),
        "test": (X_test, y_test),
    }
