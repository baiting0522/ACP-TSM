from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acp_tsm.method import ACPTSM, ACPConfig
from acp_tsm.utils.seeding import set_seed
from acp_tsm.utils.io import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    setups = [
        ("electricity", "MCAR"), ("electricity", "MAR"),
        ("oil", "MCAR"), ("oil", "MAR"),
        ("energy", "MCAR"), ("energy", "MAR"),
        ("air_quality", "MCAR"),
    ]
    for dataset, mech in setups:
        for seed in range(args.repetitions):
            set_seed(seed)
            cfg = ACPConfig(
                dataset=dataset,
                data_dir=args.data_dir,
                missing_mechanism=mech,
                random_seed=seed,
                device=args.device,
            )
            pred_df, metrics = ACPTSM(cfg).run()
            rows.append({"dataset": dataset, "mechanism": mech, "seed": seed, **metrics})
            print(dataset, mech, seed, metrics)
    df = pd.DataFrame(rows)
    out_dir = ensure_dir(ROOT / "outputs")
    df.to_csv(out_dir / "benchmark_results.csv", index=False)
    summary = df.groupby(["dataset", "mechanism"]).agg(["mean", "std"])
    summary.to_csv(out_dir / "benchmark_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
