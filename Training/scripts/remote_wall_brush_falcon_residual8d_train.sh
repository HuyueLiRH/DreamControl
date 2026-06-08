#!/usr/bin/env bash
set -euo pipefail

TASK="Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-FalconResidual8D-v0"
NUM_ENVS="${1:-1024}"
MAX_ITERATIONS="${2:-5}"
RUN_NAME="${3:-falcon_residual8d_short_smoke}"

REF_MOTIONS_PATH="../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"

source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab
cd /root/autodl-tmp/IsaacLab

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/isaac_utils:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"

RESUME_ARGS=()

TERM=xterm python scripts/reinforcement_learning/rsl_rl/train.py \
  --task="$TASK" \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  "${RESUME_ARGS[@]}" \
  --run_name "$RUN_NAME" \
  env.episode_length_s=10.0 \
  env.ref_motions_path="$REF_MOTIONS_PATH"
