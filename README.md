# Anonymous ACP-TSM reproduction code

This repository is a clean, anonymous implementation of ACP-TSM for time series conformal prediction with missing covariates.

It follows the article's main ingredients:
- iterative imputation on the training split only
- quantile regression on imputed features concatenated with masks
- subset-matched conformal calibration for missingness patterns
- post-training BiLSTM correction with dual loss
- optional adaptive fine-tuning at test time

Included datasets from the original project archive:
- `electricity.csv`
- `ETTm1.csv`
- `energydata_complete.csv`
- `AirQualityUCI.xlsx`

## Install

```bash
pip install -r requirements.txt
```

## Run one dataset

```bash
python scripts/run_dataset.py --dataset electricity --mechanism MCAR --missing-rate 0.4
python scripts/run_dataset.py --dataset oil --mechanism MAR --missing-rate 0.4
python scripts/run_dataset.py --dataset energy --mechanism MNAR --missing-rate 0.4
python scripts/run_dataset.py --dataset air_quality
```

## Run the benchmark table

```bash
python scripts/run_benchmark.py --repetitions 5
```

Outputs are written to `outputs/`.
