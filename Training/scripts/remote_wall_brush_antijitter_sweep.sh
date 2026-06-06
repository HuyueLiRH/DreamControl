#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?checkpoint path is required}"
OUTPUT_DIR="${2:-/root/autodl-tmp/wall_brush_antijitter_sweep}"
ALPHAS="${3:-1.00 0.85 0.70 0.55}"
NUM_ENVS="${4:-27}"
NUM_STEPS="${5:-500}"
EVAL_WRAPPER="$(dirname "$0")/remote_wall_brush_buttonpress_aligned_antijitter_eval.sh"

mkdir -p "$OUTPUT_DIR"

for ACTION_SMOOTHING_ALPHA in $ALPHAS; do
  ALPHA_TAG="${ACTION_SMOOTHING_ALPHA/./p}"
  OUTPUT="$OUTPUT_DIR/eval_alpha_${ALPHA_TAG}.json"
  "$EVAL_WRAPPER" "$CHECKPOINT" "$NUM_ENVS" "$NUM_STEPS" "$OUTPUT" 0 500 0 "$ACTION_SMOOTHING_ALPHA"
done
