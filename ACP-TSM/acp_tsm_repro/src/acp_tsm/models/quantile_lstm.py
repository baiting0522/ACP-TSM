from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class QuantileLSTM(nn.Module):
    def __init__(self, input_dim: int, extra_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, sequence: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(sequence)
        h = out[:, -1, :]
        pred = self.mlp(torch.cat([h, extras], dim=1))
        q_low = pred[:, :1]
        q_high = pred[:, 1:2]
        lo = torch.minimum(q_low, q_high)
        hi = torch.maximum(q_low, q_high)
        return torch.cat([lo, hi], dim=1)


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, q: float) -> torch.Tensor:
    err = y_true - y_pred
    return torch.mean(torch.maximum(q * err, (q - 1.0) * err))


@dataclass
class QuantileTrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    alpha: float = 0.1
    device: str = "cpu"


def fit_quantile_model(model: QuantileLSTM, train_ds, cfg: QuantileTrainConfig) -> QuantileLSTM:
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    ql, qh = cfg.alpha / 2.0, 1.0 - cfg.alpha / 2.0
    for _ in range(cfg.epochs):
        model.train()
        for batch in loader:
            seq = batch["sequence"].to(cfg.device)
            extras = batch["extras"].to(cfg.device)
            y = batch["target"].to(cfg.device).unsqueeze(1)
            pred = model(seq, extras)
            loss = pinball_loss(y, pred[:, :1], ql) + pinball_loss(y, pred[:, 1:2], qh)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
