#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${1:-256}"
MAX_ITERATIONS="${2:-1}"
RUN_NAME="${3:-falcon_residual8d_smoke}"
ZERO_OUTPUT="${4:-/root/autodl-tmp/wall_brush_falcon_residual8d_zero_smoke.json}"

TASK="Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-FalconResidual8D-v0"
REF_MOTIONS_PATH="../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"
REF_MOTIONS_ABS="/root/autodl-tmp/TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"
EPISODE_LENGTH_OVERRIDE="env.episode_length_s=10.0"

source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab
cd /root/autodl-tmp/IsaacLab

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/isaac_utils:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"

if ps -eo pid,ppid,stat,etime,cmd | grep -E "[i]saac|[k]it|[t]rain.py|[p]lay.py|[e]val_wall_brush" >/tmp/wall_brush_active_jobs.txt; then
  cat /tmp/wall_brush_active_jobs.txt
  echo "Refusing to start FALCON smoke while another Isaac/Kit train/eval/video job is active." >&2
  exit 2
fi

python -m unittest source/isaaclab_tasks/test/test_wall_brush_success.py source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v

python - <<'PY'
from pathlib import Path
import numpy as np

path = Path("/root/autodl-tmp/TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz")
if not path.exists():
    raise SystemExit(f"missing staged250 prior: {path}")
data = np.load(path, allow_pickle=True)
motion_arrays = [
    value for value in (data[key] for key in data.files)
    if hasattr(value, "ndim") and value.ndim >= 2 and value.shape[:2] == (27, 250)
]
if not motion_arrays:
    raise SystemExit(f"no array with leading shape (27, 250) in {path}; keys={sorted(data.files)}")
print("staged250 prior ok: motions=27 frames=250")
PY

bash scripts/remote_wall_brush_falcon_residual8d_eval.sh 27 500 "$ZERO_OUTPUT" 0 1 500 1.0
bash scripts/remote_wall_brush_falcon_residual8d_train.sh "$NUM_ENVS" "$MAX_ITERATIONS" "$RUN_NAME"

printf 'FALCON smoke finished with %s and %s\n' "$REF_MOTIONS_PATH" "$EPISODE_LENGTH_OVERRIDE"
