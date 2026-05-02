from __future__ import annotations
import math
import numpy as np


def subset_match_indices(mask_cal: np.ndarray, mask_test: np.ndarray) -> np.ndarray:
    ok = np.all((mask_cal & mask_test) == mask_cal, axis=1)
    return np.where(ok)[0]


def temporal_weights(n: int, decay: float = 0.98) -> np.ndarray:
    idx = np.arange(n)
    w = decay ** (n - 1 - idx)
    return w / w.sum()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, q, side="left")])


def nonconformity_scores(y: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> np.ndarray:
    return np.maximum(q_low - y, y - q_high)


def split_cp_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def pattern_matched_quantile(cal_scores: np.ndarray, cal_mask: np.ndarray, test_mask: np.ndarray, alpha: float, decay: float = 0.98) -> float:
    idx = subset_match_indices(cal_mask.astype(int), test_mask.astype(int))
    if len(idx) == 0:
        idx = np.arange(len(cal_scores))
    scores = cal_scores[idx]
    weights = temporal_weights(len(scores), decay=decay)
    return weighted_quantile(scores, weights, 1 - alpha)
