from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acp_tsm.method import ACPTSM, ACPConfig
from acp_tsm.utils.seeding import set_seed
from acp_tsm.utils.io import ensure_dir, dump_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["electricity", "oil", "energy", "air_quality"], required=True)
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--mechanism", default="MCAR", choices=["MCAR", "MAR", "MNAR"])
    p.add_argument("--missing-rate", type=float, default=0.4)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = ACPConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        alpha=args.alpha,
        missing_mechanism=args.mechanism,
        missing_rate=args.missing_rate,
        random_seed=args.seed,
        lookback=args.lookback,
        device=args.device,
    )
    method = ACPTSM(cfg)
    pred_df, metrics = method.run()
    out_dir = ensure_dir(ROOT / "outputs" / args.dataset)
    tag = f"{args.dataset}_{args.mechanism.lower()}_seed{args.seed}"
    pred_df.to_csv(out_dir / f"{tag}_predictions.csv", index=False)
    dump_json({"config": method.config_dict(), "metrics": metrics}, out_dir / f"{tag}_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
