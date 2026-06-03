#!/usr/bin/env bash
set -euo pipefail

TASK="Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAligned-v0"
NUM_ENVS="${1:-16}"
MAX_ITERATIONS="${2:-1}"
RUN_NAME="${3:-buttonpress_aligned_main_smoke}"

REF_MOTIONS_PATH="../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"
REF_MOTIONS_ABS="/root/autodl-tmp/TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"

source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab
cd /root/autodl-tmp/IsaacLab

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/isaac_utils:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"

if ps -eo pid,ppid,stat,etime,cmd | grep -E "[i]saac|[k]it|[t]rain.py|[p]lay.py|[e]val_wall_brush" >/tmp/wall_brush_active_jobs.txt; then
  cat /tmp/wall_brush_active_jobs.txt
  echo "Refusing to start smoke while another Isaac/Kit train/eval/video job is active." >&2
  exit 2
fi

python -m unittest source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v

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

def scalar(name, default):
    if name not in data.files:
        return default
    return float(np.asarray(data[name]).reshape(-1)[0])

fps = scalar("fps", 25.0)
episode_length_s = scalar("episode_length_s", 10.0)
control_steps = int(round(scalar("control_steps_per_episode", episode_length_s * 50.0)))

assert len(motion_arrays) > 0
assert abs(fps - 25.0) < 1e-6, fps
assert abs(episode_length_s - 10.0) < 1e-6, episode_length_s
assert control_steps == 500, control_steps
print(
    "staged250 prior ok: "
    f"motions=27 frames=250 fps={fps} episode_length_s={episode_length_s} control_steps={control_steps}"
)
PY

TERM=xterm python scripts/reinforcement_learning/rsl_rl/train.py \
  --task="$TASK" \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --resume_action_std 0.003 \
  --run_name "$RUN_NAME" \
  env.episode_length_s=10.0 \
  env.ref_motions_path="$REF_MOTIONS_PATH"
