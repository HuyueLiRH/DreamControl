#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-FalconResidual8D-v0}"
NUM_ENVS="${1:-1024}"
MAX_ITERATIONS="${2:-5}"
RUN_NAME="${3:-workstation_falcon_residual8d_short_smoke}"

source /project/huyue/src/DreamControl/scripts/workstation_wall_brush_env.sh
cd "$DREAMCONTROL_ROOT/Training"

TERM=xterm python scripts/reinforcement_learning/rsl_rl/train.py \
  --task="$TASK" \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --run_name "$RUN_NAME" \
  env.episode_length_s=10.0 \
  env.ref_motions_path="$WALL_BRUSH_PRIOR"
