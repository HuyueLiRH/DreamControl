#!/usr/bin/env bash
set -euo pipefail

TASK="Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-FalconResidual8D-v0"
NUM_ENVS="${1:-27}"
NUM_STEPS="${2:-500}"
OUTPUT="${3:-/root/autodl-tmp/wall_brush_falcon_residual8d_zero_eval.json}"
VISUAL_REVIEW="${4:-0}"
ZERO_ACTIONS="${5:-1}"
VIDEO_LENGTH="${6:-500}"
ACTION_SMOOTHING_ALPHA="${7:-1.0}"

REF_MOTIONS_PATH="../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"

source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab
cd /root/autodl-tmp/IsaacLab

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/isaac_utils:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"

EXTRA_EVAL_ARGS=(--skip_checkpoint)
if [[ "$ZERO_ACTIONS" != "0" ]]; then
  EXTRA_EVAL_ARGS+=(--zero_actions)
fi

TERM=xterm python scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py \
  --task="$TASK" \
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
  echo "Visual review for skip-checkpoint FALCON zero-residual eval is deferred until smoke metrics pass." >&2
  echo "Requested video length: $VIDEO_LENGTH" >&2
fi
