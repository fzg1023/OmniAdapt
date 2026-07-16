#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/setup_paths.sh <data_dir> [save_dir]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$1"
SAVE_DIR="${2:-$ROOT_DIR/output}"

cd "$ROOT_DIR"
python tracking/create_default_local_file.py \
  --workspace_dir "$ROOT_DIR" \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR"
