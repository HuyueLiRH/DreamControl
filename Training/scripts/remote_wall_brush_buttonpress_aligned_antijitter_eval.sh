#!/usr/bin/env bash
set -euo pipefail

TASK="Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-v0"
CHECKPOINT="${1:?checkpoint path is required}"
NUM_ENVS="${2:-27}"
NUM_STEPS="${3:-500}"
OUTPUT="${4:-/root/autodl-tmp/wall_brush_buttonpress_aligned_antijitter_eval.json}"
VISUAL_REVIEW="${5:-0}"
VIDEO_LENGTH="${6:-500}"
ZERO_ACTIONS="${7:-0}"
ACTION_SMOOTHING_ALPHA="${8:-1.0}"

REF_MOTIONS_PATH="../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"

source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab
cd /root/autodl-tmp/IsaacLab

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/isaac_utils:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"

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
  env.ref_motions_path="$REF_MOTIONS_PATH"

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
    env.ref_motions_path="$REF_MOTIONS_PATH"

  TERM=xterm python scripts/reinforcement_learning/rsl_rl/play_wall_brush_fixed_view.py \
    --task="$TASK" \
    --checkpoint="$CHECKPOINT" \
    --headless \
    --device cuda:0 \
    --num_envs 1 \
    --video \
    --video_length "$VIDEO_LENGTH" \
    --action_smoothing_alpha "$ACTION_SMOOTHING_ALPHA" \
    --camera_eye="-0.85,-1.65,1.22" \
    --camera_lookat="0.36,0.00,0.92" \
    --camera_resolution="1600,1000" \
    --view_name="${VIEW_PREFIX}_prior${PRIOR_ID}_oblique_wall_marker" \
    --prior_id "$PRIOR_ID" \
    env.episode_length_s=10.0 \
    env.ref_motions_path="$REF_MOTIONS_PATH"

  python scripts/reinforcement_learning/rsl_rl/wall_brush_eval_visual_review.py \
    --eval_json "$OUTPUT" \
    --checkpoint "$CHECKPOINT" \
    --view_prefix "$VIEW_PREFIX" \
    --video_length "$VIDEO_LENGTH" \
    --update
fi
