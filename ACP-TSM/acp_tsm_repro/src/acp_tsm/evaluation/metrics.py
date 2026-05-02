from __future__ import annotations
import numpy as np


def empirical_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y >= lower) & (y <= upper)))


def average_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def winkler_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    width = upper - lower
    below = y < lower
    above = y > upper
    score = width.copy()
    score[below] += (2 / alpha) * (lower[below] - y[below])
    score[above] += (2 / alpha) * (y[above] - upper[above])
    return float(score.mean())


def coverage_gap(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, groups: np.ndarray, alpha: float) -> float:
    gaps = []
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() == 0:
            continue
        cov = empirical_coverage(y[mask], lower[mask], upper[mask])
        gaps.append(abs(cov - (1 - alpha)))
    return float(np.mean(gaps)) if gaps else 0.0


def summarize_metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, groups: np.ndarray, alpha: float) -> dict:
    return {
        "coverage": empirical_coverage(y, lower, upper),
        "width": average_width(lower, upper),
        "cov_gap": coverage_gap(y, lower, upper, groups, alpha),
        "winkler": winkler_score(y, lower, upper, alpha),
    }
