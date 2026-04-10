#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PROJECT_ROOT
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/artifacts/checkpoints}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/outputs}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/src/foundation:$PROJECT_ROOT/src/generation:${PYTHONPATH:-}"

cd "$PROJECT_ROOT/src/foundation/linearprobing"
bash test.sh "$@"

