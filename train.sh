#!/usr/bin/env bash
set -euo pipefail

python scripts/preprocess_data.py --config configs/train.yaml
python scripts/train.py --config configs/train.yaml --train-file sample_data/train.csv
