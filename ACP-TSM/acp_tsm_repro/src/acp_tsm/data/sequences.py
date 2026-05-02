from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_mask_features(mask_window: np.ndarray) -> np.ndarray:
    # mask_window: [lookback, d]
    mean = mask_window.mean(axis=0)
    std = mask_window.std(axis=0)
    jumps = np.abs(np.diff(mask_window, axis=0)).sum(axis=0) / max(1, mask_window.shape[0] - 1)
    total_missing = np.array([mask_window[-1].sum()], dtype=float)
    return np.concatenate([mean, std, jumps, total_missing], axis=0)


class SequenceDataset(Dataset):
    def __init__(self, X_imp: np.ndarray, mask: np.ndarray, y: np.ndarray, lookback: int):
        self.X_imp = X_imp.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.y = y.astype(np.float32)
        self.lookback = lookback
        self.indices = np.arange(lookback, len(X_imp))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = self.indices[i]
        x_seq = self.X_imp[t - self.lookback:t]
        m_seq = self.mask[t - self.lookback:t]
        seq = np.concatenate([x_seq, m_seq], axis=1)
        extras = compute_mask_features(m_seq)
        return {
            "sequence": torch.tensor(seq, dtype=torch.float32),
            "extras": torch.tensor(extras, dtype=torch.float32),
            "target": torch.tensor(self.y[t], dtype=torch.float32),
            "index": t,
        }
