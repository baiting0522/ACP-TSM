from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class BiLSTMCorrector(nn.Module):
    def __init__(self, input_dim: int, extra_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_dim * 2 + extra_dim, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, sequence: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(sequence)
        h = out[:, -1, :]
        h = torch.cat([h, extras], dim=1)
        h = self.dropout(self.act(self.fc1(h)))
        delta = self.fc2(h)
        return delta


class CorrectionDataset(Dataset):
    def __init__(self, sequence, extras, y, base_lower, base_upper, group_ids):
        self.sequence = torch.tensor(sequence, dtype=torch.float32)
        self.extras = torch.tensor(extras, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.base_lower = torch.tensor(base_lower, dtype=torch.float32)
        self.base_upper = torch.tensor(base_upper, dtype=torch.float32)
        self.group_ids = torch.tensor(group_ids, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            "sequence": self.sequence[idx],
            "extras": self.extras[idx],
            "y": self.y[idx],
            "base_lower": self.base_lower[idx],
            "base_upper": self.base_upper[idx],
            "group_id": self.group_ids[idx],
        }


@dataclass
class CorrectorTrainConfig:
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    alpha: float = 0.1
    lambda_eff: float = 0.1
    smooth_c: float = 10.0
    device: str = "cpu"


def _batch_dual_loss(delta, base_lower, base_upper, y, group_ids, alpha, lambda_eff, smooth_c):
    lower = base_lower - delta[:, 0]
    upper = base_upper + delta[:, 1]
    lower_cov = torch.sigmoid(smooth_c * (y - lower))
    upper_cov = torch.sigmoid(smooth_c * (upper - y))
    covered = lower_cov * upper_cov

    cov_terms = []
    for gid in torch.unique(group_ids):
        mask = group_ids == gid
        smooth_coverage = covered[mask].mean()
        cov_terms.append(torch.abs(smooth_coverage - (1 - alpha)))
    l_cov = torch.stack(cov_terms).max() if cov_terms else torch.tensor(0.0, device=y.device)

    width = upper - lower
    l_eff = width.mean()
    l_reg = ((upper - base_upper) ** 2 + (lower - base_lower) ** 2).mean()
    total = l_cov + l_reg + lambda_eff * l_eff
    return total, {"cov": l_cov.detach(), "eff": l_eff.detach(), "reg": l_reg.detach()}


def fit_corrector(model: BiLSTMCorrector, train_ds: CorrectionDataset, cfg: CorrectorTrainConfig) -> BiLSTMCorrector:
    model.to(cfg.device)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        model.train()
        for batch in loader:
            seq = batch["sequence"].to(cfg.device)
            extras = batch["extras"].to(cfg.device)
            y = batch["y"].to(cfg.device)
            base_lower = batch["base_lower"].to(cfg.device)
            base_upper = batch["base_upper"].to(cfg.device)
            group_ids = batch["group_id"].to(cfg.device)
            delta = model(seq, extras)
            loss, _ = _batch_dual_loss(delta, base_lower, base_upper, y, group_ids, cfg.alpha, cfg.lambda_eff, cfg.smooth_c)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
