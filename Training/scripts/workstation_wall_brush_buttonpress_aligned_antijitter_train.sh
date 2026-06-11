#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-v0}"
NUM_ENVS="${1:-8192}"
MAX_ITERATIONS="${2:-300}"
LOAD_RUN="${3:-}"
CHECKPOINT="${4:-}"
RUN_NAME="${5:-workstation_buttonpress_aligned_antijitter_finetune}"
RESUME_ACTION_STD="${6:-0.0015}"

source /project/huyue/src/DreamControl/scripts/workstation_wall_brush_env.sh
cd "$DREAMCONTROL_ROOT/Training"

RESUME_ARGS=()
if [[ -n "$LOAD_RUN" && -n "$CHECKPOINT" ]]; then
  RESUME_ARGS+=(--resume --load_run "$LOAD_RUN" --checkpoint "$CHECKPOINT" --reset_optimizer_on_resume)
fi

TERM=xterm python scripts/reinforcement_learning/rsl_rl/train.py \
  --task="$TASK" \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --resume_action_std "$RESUME_ACTION_STD" \
  "${RESUME_ARGS[@]}" \
  --run_name "$RUN_NAME" \
  env.episode_length_s=10.0 \
  env.ref_motions_path="$WALL_BRUSH_PRIOR"
