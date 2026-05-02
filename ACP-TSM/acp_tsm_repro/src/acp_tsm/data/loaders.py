from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str


def load_dataset(name: str, data_dir: str | Path) -> DatasetBundle:
    name = name.lower()
    data_dir = Path(data_dir)
    if name == "electricity":
        return load_electricity(data_dir / "electricity.csv")
    if name == "oil":
        return load_oil(data_dir / "ETTm1.csv")
    if name == "energy":
        return load_energy(data_dir / "energydata_complete.csv")
    if name == "air_quality":
        return load_air_quality(data_dir / "AirQualityUCI.xlsx")
    raise ValueError(f"Unknown dataset: {name}")


def load_electricity(path: str | Path) -> DatasetBundle:
    df = pd.read_csv(path)
    df = df.iloc[17760:].reset_index(drop=True)
    period = df["period"].to_numpy()
    keep = (period > period[17]) & (period < period[24])
    df = df.loc[keep].reset_index(drop=True)
    features = ["nswprice", "nswdemand", "vicprice", "vicdemand"]
    target = "transfer"
    X = df[features].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    return DatasetBundle("electricity", X, y, features, target)


def load_oil(path: str | Path) -> DatasetBundle:
    df = pd.read_csv(path).iloc[:8000].reset_index(drop=True)
    features = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
    target = "OT"
    X = df[features].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    return DatasetBundle("oil", X, y, features, target)


def load_energy(path: str | Path) -> DatasetBundle:
    df = pd.read_csv(path)
    target = "Appliances"
    features = [c for c in df.columns if c != target]
    X = df[features].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    return DatasetBundle("energy", X, y, features, target)


def load_air_quality(path: str | Path) -> DatasetBundle:
    df = pd.read_excel(path).replace(-200, np.nan)
    target = "CO(GT)"
    features = [c for c in df.columns if c not in [target, "Date", "Time"]]
    df = df.dropna(subset=[target]).reset_index(drop=True)
    missing_count = df[features].isna().sum(axis=1)
    df = df.loc[missing_count < len(features)].reset_index(drop=True)
    X = df[features].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    return DatasetBundle("air_quality", X, y, features, target)
