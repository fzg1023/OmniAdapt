#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "Usage: bash scripts/test.sh <checkpoint> <dataset> <data_root> [epoch] [workers] [config]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="$1"
DATASET="$2"
DATA_ROOT="$3"
EPOCH="${4:-12}"
WORKERS="${5:-1}"
CONFIG="${6:-std_full}"

cd "$ROOT_DIR"
python tracking/test.py \
  --yaml_name "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET" \
  --data_root "$DATA_ROOT" \
  --epoch "$EPOCH" \
  --workers "$WORKERS"
