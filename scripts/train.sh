#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-std_full}"
MODE="${2:-single}"
NUM_GPUS="${3:-1}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/output}"

if [[ "$MODE" != "single" && "$MODE" != "multiple" && "$MODE" != "multi_node" ]]; then
  echo "MODE must be one of: single, multiple, multi_node" >&2
  exit 1
fi

cd "$ROOT_DIR"
python tracking/train.py \
  --script omniadapt \
  --config "$CONFIG" \
  --mode "$MODE" \
  --nproc_per_node "$NUM_GPUS" \
  --save_dir "$SAVE_DIR"
