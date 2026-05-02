from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data.loaders import load_dataset
from .data.missingness import apply_missingness
from .data.splits import chronological_split
from .data.sequences import SequenceDataset, compute_mask_features
from .models.quantile_lstm import QuantileLSTM, QuantileTrainConfig, fit_quantile_model
from .models.corrector import BiLSTMCorrector, CorrectionDataset, CorrectorTrainConfig, fit_corrector
from .conformal.base import nonconformity_scores, split_cp_quantile, pattern_matched_quantile
from .evaluation.metrics import summarize_metrics


@dataclass
class ACPConfig:
    dataset: str
    data_dir: str = "data/raw"
    alpha: float = 0.1
    missing_mechanism: str = "MCAR"
    missing_rate: float = 0.4
    lookback: int = 24
    train_cal_cut: int | None = None
    test_size: int | None = None
    random_seed: int = 0
    correction_fraction: float = 0.3
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    q_epochs: int = 20
    c_epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    lambda_eff: float = 0.1
    smooth_c: float = 10.0
    adaptive_eta: float = 0.005
    adaptive: bool = True
    device: str = "cpu"


class ACPTSM:
    def __init__(self, cfg: ACPConfig):
        self.cfg = cfg
        self.imputer = IterativeImputer(estimator=Ridge(), max_iter=10, random_state=cfg.random_seed)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.quantile_model = None
        self.corrector = None
        self.fitted = False

    def _prepare_raw(self):
        bundle = load_dataset(self.cfg.dataset, self.cfg.data_dir)
        X, y = bundle.X.copy(), bundle.y.copy()
        if bundle.name != "air_quality":
            X, synthetic_mask = apply_missingness(X, self.cfg.missing_mechanism, self.cfg.missing_rate, self.cfg.random_seed)
        else:
            synthetic_mask = np.isnan(X).astype(int)
        return bundle, X, y, synthetic_mask

    def _split(self, X, y):
        if self.cfg.dataset == "air_quality":
            split = chronological_split(X, y, train_cal_cut=5000, test_size=300, seed=self.cfg.random_seed)
        elif self.cfg.dataset == "energy":
            split = chronological_split(X, y, train_cal_cut=5000, test_size=None, seed=self.cfg.random_seed)
        else:
            split = chronological_split(X, y, train_cal_cut=self.cfg.train_cal_cut, test_size=self.cfg.test_size, seed=self.cfg.random_seed)
        return split

    def _fit_imputer_and_scalers(self, X_train, y_train):
        X_train_imp = self.imputer.fit_transform(X_train)
        self.scaler_x.fit(X_train_imp)
        self.scaler_y.fit(y_train.reshape(-1, 1))

    def _transform_features(self, X):
        mask = np.isnan(X).astype(np.float32)
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler_x.transform(X_imp)
        return X_scaled.astype(np.float32), mask.astype(np.float32)

    def _transform_target(self, y):
        return self.scaler_y.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    def fit(self):
        bundle, X_raw, y_raw, _ = self._prepare_raw()
        split = self._split(X_raw, y_raw)
        X_train, y_train = split["train"]
        X_cal, y_cal = split["cal"]
        X_test, y_test = split["test"]

        self._fit_imputer_and_scalers(X_train, y_train)
        X_train_s, mask_train = self._transform_features(X_train)
        X_cal_s, mask_cal = self._transform_features(X_cal)
        X_test_s, mask_test = self._transform_features(X_test)
        y_train_s = self._transform_target(y_train)
        y_cal_s = self._transform_target(y_cal)
        y_test_s = self._transform_target(y_test)

        train_ds = SequenceDataset(X_train_s, mask_train, y_train_s, self.cfg.lookback)
        seq_input_dim = X_train_s.shape[1] * 2
        extra_dim = compute_mask_features(mask_train[: self.cfg.lookback]).shape[0]
        self.quantile_model = QuantileLSTM(seq_input_dim, extra_dim, self.cfg.hidden_dim, self.cfg.num_layers, self.cfg.dropout)
        self.quantile_model = fit_quantile_model(
            self.quantile_model,
            train_ds,
            QuantileTrainConfig(
                epochs=self.cfg.q_epochs,
                batch_size=self.cfg.batch_size,
                lr=self.cfg.lr,
                alpha=self.cfg.alpha,
                device=self.cfg.device,
            ),
        )

        cal_pred = self._predict_base(X_cal_s, mask_cal, y_cal_s)
        q_base = split_cp_quantile(cal_pred["scores"], self.cfg.alpha)
        cal_base_lower = cal_pred["q_low"] - q_base
        cal_base_upper = cal_pred["q_high"] + q_base

        corr_n = max(32, int(len(cal_pred["y_seq"]) * self.cfg.correction_fraction))
        corr_seq = cal_pred["sequence"][:corr_n]
        corr_extras = cal_pred["extras"][:corr_n]
        corr_y = cal_pred["y_seq"][:corr_n]
        corr_lower = cal_base_lower[:corr_n]
        corr_upper = cal_base_upper[:corr_n]
        corr_groups = cal_pred["group_ids"][:corr_n]
        self.corrector = BiLSTMCorrector(seq_input_dim, corr_extras.shape[1], self.cfg.hidden_dim, self.cfg.num_layers, self.cfg.dropout)
        corr_ds = CorrectionDataset(corr_seq, corr_extras, corr_y, corr_lower, corr_upper, corr_groups)
        self.corrector = fit_corrector(
            self.corrector,
            corr_ds,
            CorrectorTrainConfig(
                epochs=self.cfg.c_epochs,
                batch_size=self.cfg.batch_size,
                lr=self.cfg.lr,
                alpha=self.cfg.alpha,
                lambda_eff=self.cfg.lambda_eff,
                smooth_c=self.cfg.smooth_c,
                device=self.cfg.device,
            ),
        )

        self.cache = {
            "bundle": bundle,
            "X_train_s": X_train_s, "mask_train": mask_train, "y_train": y_train, "y_train_s": y_train_s,
            "X_cal_s": X_cal_s, "mask_cal": mask_cal, "y_cal": y_cal, "y_cal_s": y_cal_s,
            "X_test_s": X_test_s, "mask_test": mask_test, "y_test": y_test, "y_test_s": y_test_s,
            "q_base": q_base,
        }
        self.fitted = True
        return self

    @torch.no_grad()
    def _predict_base(self, X_s, mask, y_s):
        self.quantile_model.eval()
        seqs, extras, ql, qh, ys, gids = [], [], [], [], [], []
        for t in range(self.cfg.lookback, len(X_s)):
            x_seq = X_s[t - self.cfg.lookback:t]
            m_seq = mask[t - self.cfg.lookback:t]
            seq = np.concatenate([x_seq, m_seq], axis=1).astype(np.float32)
            extra = compute_mask_features(m_seq).astype(np.float32)
            seq_t = torch.tensor(seq[None, ...], dtype=torch.float32, device=self.cfg.device)
            extra_t = torch.tensor(extra[None, ...], dtype=torch.float32, device=self.cfg.device)
            pred = self.quantile_model(seq_t, extra_t).cpu().numpy()[0]
            ql.append(pred[0]); qh.append(pred[1])
            seqs.append(seq); extras.append(extra); ys.append(y_s[t]); gids.append(int(mask[t].sum()))
        ql = np.array(ql); qh = np.array(qh); ys = np.array(ys)
        scores = nonconformity_scores(ys, ql, qh)
        return {
            "sequence": np.stack(seqs),
            "extras": np.stack(extras),
            "q_low": ql,
            "q_high": qh,
            "scores": scores,
            "y_seq": ys,
            "group_ids": np.array(gids),
        }

    @torch.no_grad()
    def predict(self):
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        X_cal_s = self.cache["X_cal_s"]
        mask_cal = self.cache["mask_cal"]
        y_cal_s = self.cache["y_cal_s"]
        X_test_s = self.cache["X_test_s"]
        mask_test = self.cache["mask_test"]
        y_test = self.cache["y_test"]
        y_test_s = self.cache["y_test_s"]

        cal_pred = self._predict_base(X_cal_s, mask_cal, y_cal_s)
        test_pred = self._predict_base(X_test_s, mask_test, y_test_s)
        cal_scores = cal_pred["scores"]
        cal_masks_eff = mask_cal[self.cfg.lookback:]

        self.corrector.eval()
        lower_final, upper_final = [], []
        alpha_t = self.cfg.alpha
        for i in range(len(test_pred["y_seq"])):
            q_hat = pattern_matched_quantile(cal_scores, cal_masks_eff, mask_test[self.cfg.lookback + i], alpha_t)
            base_lower = test_pred["q_low"][i] - q_hat
            base_upper = test_pred["q_high"][i] + q_hat
            seq_t = torch.tensor(test_pred["sequence"][i:i+1], dtype=torch.float32, device=self.cfg.device)
            extra_t = torch.tensor(test_pred["extras"][i:i+1], dtype=torch.float32, device=self.cfg.device)
            delta = self.corrector(seq_t, extra_t).cpu().numpy()[0]
            lo = base_lower - delta[0]
            hi = base_upper + delta[1]
            if hi < lo:
                mid = 0.5 * (hi + lo)
                rad = 0.5 * abs(hi - lo)
                lo, hi = mid - rad, mid + rad
            lower_final.append(lo)
            upper_final.append(hi)
            if self.cfg.adaptive:
                covered = float(lo <= test_pred["y_seq"][i] <= hi)
                alpha_t = float(np.clip(alpha_t + self.cfg.adaptive_eta * (self.cfg.alpha - (1 - covered)), 0.001, 0.999))

        lower_final = np.array(lower_final)
        upper_final = np.array(upper_final)
        y_eval = y_test[self.cfg.lookback:]
        lower = self.scaler_y.inverse_transform(lower_final.reshape(-1, 1)).reshape(-1)
        upper = self.scaler_y.inverse_transform(upper_final.reshape(-1, 1)).reshape(-1)
        groups = mask_test[self.cfg.lookback:].sum(axis=1).astype(int)
        metrics = summarize_metrics(y_eval, lower, upper, groups, self.cfg.alpha)
        pred_df = pd.DataFrame({
            "y_true": y_eval,
            "lower": lower,
            "upper": upper,
            "group_missing_count": groups,
        })
        return pred_df, metrics

    def run(self):
        self.fit()
        return self.predict()

    def config_dict(self):
        return asdict(self.cfg)
