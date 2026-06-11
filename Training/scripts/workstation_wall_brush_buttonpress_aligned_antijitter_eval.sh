#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-v0}"
CHECKPOINT="${1:-/project/huyue/checkpoints/wall_brush/buttonpress_aligned_antijitter_model_1999.pt}"
NUM_ENVS="${2:-27}"
NUM_STEPS="${3:-500}"
OUTPUT="${4:-/project/huyue/logs/wall_brush/buttonpress_aligned_antijitter_eval.json}"
VISUAL_REVIEW="${5:-0}"
VIDEO_LENGTH="${6:-500}"
ZERO_ACTIONS="${7:-0}"
ACTION_SMOOTHING_ALPHA="${8:-1.0}"

source /project/huyue/src/DreamControl/scripts/workstation_wall_brush_env.sh
cd "$DREAMCONTROL_ROOT/Training"
mkdir -p "$(dirname "$OUTPUT")"

EXTRA_EVAL_ARGS=()
if [[ "$ZERO_ACTIONS" != "0" ]]; then
  EXTRA_EVAL_ARGS+=(--zero_actions)
fi

TERM=xterm python scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py \
  --task="$TASK" \
  --checkpoint="$CHECKPOINT" \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --num_steps "$NUM_STEPS" \
  --output "$OUTPUT" \
  --action_smoothing_alpha "$ACTION_SMOOTHING_ALPHA" \
  "${EXTRA_EVAL_ARGS[@]}" \
  env.episode_length_s=10.0 \
  env.ref_motions_path="$WALL_BRUSH_PRIOR"

if [[ "$VISUAL_REVIEW" != "0" ]]; then
  PRIOR_ID="$(python scripts/reinforcement_learning/rsl_rl/wall_brush_eval_visual_review.py --eval_json "$OUTPUT" --checkpoint "$CHECKPOINT" --print_prior)"
  VIEW_PREFIX="$(basename "${OUTPUT%.*}")"
  TERM=xterm python scripts/reinforcement_learning/rsl_rl/play_wall_brush_fixed_view.py \
    --task="$TASK" \
    --checkpoint="$CHECKPOINT" \
    --headless \
    --device cuda:0 \
    --num_envs 1 \
    --video \
    --video_length "$VIDEO_LENGTH" \
    --action_smoothing_alpha "$ACTION_SMOOTHING_ALPHA" \
    --camera_eye="0.08,-1.90,1.02" \
    --camera_lookat="0.45,0.00,0.92" \
    --camera_resolution="1600,1000" \
    --view_name="${VIEW_PREFIX}_prior${PRIOR_ID}_side_wall_marker" \
    --prior_id "$PRIOR_ID" \
    env.episode_length_s=10.0 \
    env.ref_motions_path="$WALL_BRUSH_PRIOR"
fi
