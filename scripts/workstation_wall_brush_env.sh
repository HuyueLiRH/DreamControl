#!/usr/bin/env bash
set -euo pipefail

source /project/huyue/miniconda3/etc/profile.d/conda.sh
conda activate /project/huyue/envs/dreamcontrol_train

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_ROOT=/project/huyue/src/IsaacLab
export DREAMCONTROL_ROOT=/project/huyue/src/DreamControl
export TRAJGEN_ROOT=/project/huyue/src/TrajGen
export WALL_BRUSH_PRIOR=/project/huyue/src/TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz

# Runtime uses the patched workstation IsaacLab tree. Do not put
# DreamControl/Training/source/isaaclab ahead of this path; that tree carries
# older IsaacLab app code and is incompatible with this Isaac Sim 5.1 env.
export PYTHONPATH="$DREAMCONTROL_ROOT/Training/isaac_utils:$DREAMCONTROL_ROOT/Training/source/isaaclab_assets:$ISAACLAB_ROOT/isaac_utils:$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_tasks:$ISAACLAB_ROOT/source/isaaclab_assets:$ISAACLAB_ROOT/source/isaaclab_rl:$ISAACLAB_ROOT/source/isaaclab_mimic:${PYTHONPATH:-}"
