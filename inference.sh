#!/usr/bin/env bash
set -euo pipefail

python scripts/inference.py --model-path configs/inference.yaml --message "${1:-I transferred money but the receiver has not received it.}"
