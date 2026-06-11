#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMCONTROL_ROOT="${DREAMCONTROL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/project/huyue/src/IsaacLab}"
BACKUP_ROOT="${BACKUP_ROOT:-/project/huyue/backups/isaaclab_wall_brush_overlay_$(date -u +%Y%m%dT%H%M%S)}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

if [[ ! -d "$DREAMCONTROL_ROOT/Training" ]]; then
  echo "Missing DreamControl Training directory: $DREAMCONTROL_ROOT/Training" >&2
  exit 1
fi

if [[ ! -d "$ISAACLAB_ROOT/source/isaaclab_tasks" ]]; then
  echo "Missing IsaacLab source tree: $ISAACLAB_ROOT" >&2
  exit 1
fi

mkdir -p "$BACKUP_ROOT"

backup_path() {
  local dst="$1"
  if [[ -e "$dst" || -L "$dst" ]]; then
    local rel="${dst#$ISAACLAB_ROOT/}"
    mkdir -p "$BACKUP_ROOT/$(dirname "$rel")"
    cp -a "$dst" "$BACKUP_ROOT/$rel"
  fi
}

copy_file() {
  local src="$DREAMCONTROL_ROOT/$1"
  local dst="$ISAACLAB_ROOT/$2"
  if [[ ! -f "$src" ]]; then
    echo "Missing source file: $src" >&2
    exit 1
  fi
  backup_path "$dst"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

copy_dir() {
  local src="$DREAMCONTROL_ROOT/$1"
  local dst="$ISAACLAB_ROOT/$2"
  if [[ ! -d "$src" ]]; then
    echo "Missing source directory: $src" >&2
    exit 1
  fi
  backup_path "$dst"
  mkdir -p "$(dirname "$dst")"
  rm -rf "$dst"
  cp -a "$src" "$dst"
}

if [[ "$INSTALL_DEPS" != "0" ]]; then
  python -m pip install joblib easydict loguru
fi

copy_file "Training/scripts/reinforcement_learning/rsl_rl/train.py" \
  "scripts/reinforcement_learning/rsl_rl/train.py"
copy_file "Training/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py" \
  "scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py"
copy_file "Training/scripts/reinforcement_learning/rsl_rl/play_wall_brush_fixed_view.py" \
  "scripts/reinforcement_learning/rsl_rl/play_wall_brush_fixed_view.py"
copy_file "Training/scripts/reinforcement_learning/rsl_rl/wall_brush_eval_visual_review.py" \
  "scripts/reinforcement_learning/rsl_rl/wall_brush_eval_visual_review.py"

copy_file "Training/source/isaaclab/isaaclab/envs/manager_based_rl_env.py" \
  "source/isaaclab/isaaclab/envs/manager_based_rl_env.py"
copy_file "Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/motion_tracking/g1/motion_tracking_env.py" \
  "source/isaaclab_tasks/isaaclab_tasks/manager_based/motion_tracking/g1/motion_tracking_env.py"
copy_file "Training/source/isaaclab_assets/isaaclab_assets/robots/unitree.py" \
  "source/isaaclab_assets/isaaclab_assets/robots/unitree.py"

copy_dir "Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking" \
  "source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking"
copy_dir "Training/source/isaaclab_tasks/isaaclab_tasks/utils/motion_lib" \
  "source/isaaclab_tasks/isaaclab_tasks/utils/motion_lib"

ln -sfn "$DREAMCONTROL_ROOT/Training/HumanoidVerse" "$ISAACLAB_ROOT/HumanoidVerse"

python - "$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

if "from isaaclab.assets import Articulation" not in text:
    text = text.replace("from isaaclab.envs import mdp\n", "from isaaclab.envs import mdp\nfrom isaaclab.assets import Articulation\n")

text = text.replace(
    "from isaaclab.utils.math import quat_apply_inverse, yaw_quat",
    "from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat",
)

if "def feet_contact_reward" not in text:
    text += '''


def feet_contact_reward(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward all specified feet being in contact when the command is nearly zero."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    feet_all_contact = contacts.all(dim=1)
    command = env.command_manager.get_command(command_name)
    no_command = torch.norm(command[:, :2], dim=1) < 0.06
    return (no_command & feet_all_contact).float()


def feet_parallel_to_ground(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet not being parallel to the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=body_quat.device).unsqueeze(0).repeat(body_quat.shape[0], 1)
    z_axis_w_left = quat_apply(body_quat[:, 0], z_axis)
    angle_left = torch.acos(torch.clamp(z_axis_w_left[:, 2], -1.0, 1.0))
    if body_quat.shape[1] > 1:
        z_axis_w_right = quat_apply(body_quat[:, 1], z_axis)
        angle_right = torch.acos(torch.clamp(z_axis_w_right[:, 2], -1.0, 1.0))
        return angle_left + angle_right
    return angle_left
'''

path.write_text(text)
PY

python -m py_compile \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \
  "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py" \
  "$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py" \
  "$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py" \
  "$ISAACLAB_ROOT/source/isaaclab_assets/isaaclab_assets/robots/unitree.py"

printf '%s\n' "$BACKUP_ROOT" > /project/huyue/backups/latest_isaaclab_wall_brush_workstation_overlay.txt
echo "Wall-brush workstation overlay installed."
echo "Backup: $BACKUP_ROOT"
