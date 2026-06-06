#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${NEWTON_WALL_BRUSH_CONFIG:-${ROOT_DIR}/configs/newton_wall_brush_versions.env}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config file: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CONFIG_PATH"

CHECKPOINT="${1:?checkpoint path is required}"
NUM_ENVS="${2:-27}"
NUM_STEPS="${3:-500}"
OUTPUT="${4:-$NEWTON_WALL_BRUSH_DEFAULT_OUTPUT}"
VISUAL_REVIEW="${5:-0}"
VIDEO_LENGTH="${6:-500}"
ZERO_ACTIONS="${7:-0}"

TRAINING_ROOT="${NEWTON_WALL_BRUSH_TRAINING_ROOT:-${ROOT_DIR}/Training}"
TASK="${NEWTON_WALL_BRUSH_TASK}"
REF_MOTIONS_PATH="${NEWTON_WALL_BRUSH_REF_MOTIONS_PATH}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi

if [[ -n "${NEWTON_WALL_BRUSH_CONDA_ENV:-}" ]]; then
  if [[ -f /root/miniconda3/bin/activate ]]; then
    # shellcheck source=/dev/null
    source /root/miniconda3/bin/activate "$NEWTON_WALL_BRUSH_CONDA_ENV"
  else
    echo "Conda activation requested but /root/miniconda3/bin/activate was not found." >&2
    exit 2
  fi
fi

cd "$TRAINING_ROOT"

if [[ ! -f "$REF_MOTIONS_PATH" ]]; then
  echo "Missing reference motion npz from Training root: $REF_MOTIONS_PATH" >&2
  exit 2
fi

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/isaac_utils:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic:${PYTHONPATH:-}"

EXTRA_ARGPARSE_ARGS=()
EXTRA_HYDRA_ARGS=()
if [[ -n "${NEWTON_WALL_BRUSH_PHYSICS_ARG:-}" ]]; then
  EXTRA_HYDRA_ARGS+=("$NEWTON_WALL_BRUSH_PHYSICS_ARG")
fi
if [[ "$ZERO_ACTIONS" != "0" ]]; then
  EXTRA_ARGPARSE_ARGS+=(--zero_actions)
fi

TERM=xterm python scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py \
  --task="$TASK" \
  --checkpoint="$CHECKPOINT" \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --num_steps "$NUM_STEPS" \
  --output "$OUTPUT" \
  "${EXTRA_ARGPARSE_ARGS[@]}" \
  env.episode_length_s=10.0 \
  env.ref_motions_path="$REF_MOTIONS_PATH" \
  "${EXTRA_HYDRA_ARGS[@]}"

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
    --camera_eye="0.08,-1.90,1.02" \
    --camera_lookat="0.45,0.00,0.92" \
    --camera_resolution="1600,1000" \
    --view_name="${VIEW_PREFIX}_prior${PRIOR_ID}_newton_side" \
    --prior_id "$PRIOR_ID" \
    --hide_wall_brush_markers \
    --disable_review_wall_material \
    env.episode_length_s=10.0 \
    env.ref_motions_path="$REF_MOTIONS_PATH" \
    "${EXTRA_HYDRA_ARGS[@]}"
fi
