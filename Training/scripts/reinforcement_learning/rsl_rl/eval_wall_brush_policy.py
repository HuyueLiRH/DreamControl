#!/usr/bin/env python3
"""Evaluate a wall-brush RSL-RL policy against the 27 fixed prior motions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate a wall-brush policy on deterministic prior IDs.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=27, help="Number of environments to evaluate.")
parser.add_argument("--num_steps", type=int, default=225, help="Maximum policy steps per evaluated episode.")
parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Wall-Brush-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use a published pretrained checkpoint.")
parser.add_argument("--output", type=str, default="/root/autodl-tmp/wall_brush_eval.json", help="JSON output path.")
parser.add_argument("--contact_x_threshold", type=float, default=0.03, help="Legal brush-tip contact tolerance in meters.")
parser.add_argument("--row_yz_threshold", type=float, default=0.05, help="Training-milestone single-row YZ tolerance in meters.")
parser.add_argument("--target_row_yz_threshold", type=float, default=0.03, help="Acceptance-target mean row YZ tolerance.")
parser.add_argument("--target_row_yz_p95_threshold", type=float, default=0.06, help="Acceptance-target row YZ p95 tolerance.")
parser.add_argument("--torso_wall_clearance", type=float, default=0.10, help="Training-milestone torso/pelvis/head wall clearance.")
parser.add_argument("--target_torso_wall_clearance", type=float, default=0.15, help="Acceptance-target torso/pelvis/head clearance.")
parser.add_argument("--nonbrush_wall_clearance", type=float, default=0.05, help="Training-milestone non-brush body wall clearance.")
parser.add_argument("--target_nonbrush_wall_clearance", type=float, default=0.08, help="Acceptance-target non-brush body clearance.")
parser.add_argument("--hand_wall_clearance", type=float, default=0.03, help="Training-milestone hand/wrist/forearm clearance.")
parser.add_argument("--target_hand_wall_clearance", type=float, default=0.06, help="Acceptance-target hand/wrist/forearm clearance.")
parser.add_argument("--wall_contact_force_threshold", type=float, default=1.0, help="Illegal non-brush body wall-contact force threshold in Newtons.")
parser.add_argument("--wall_contact_near_margin", type=float, default=0.22, help="Only count x-axis contact forces from bodies this close to the wall plane.")
parser.add_argument("--foot_slip_threshold", type=float, default=0.05, help="Training-milestone max active foot slip in meters.")
parser.add_argument("--target_foot_slip_threshold", type=float, default=0.03, help="Acceptance-target max active foot slip in meters.")
parser.add_argument("--torso_yaw_threshold_deg", type=float, default=35.0, help="Training-milestone torso yaw p95 threshold.")
parser.add_argument("--target_torso_yaw_threshold_deg", type=float, default=25.0, help="Acceptance-target torso yaw mean threshold.")
parser.add_argument("--torso_upright_threshold_deg", type=float, default=18.0, help="Training-milestone mean torso pitch/roll tilt threshold.")
parser.add_argument("--target_torso_upright_threshold_deg", type=float, default=12.0, help="Acceptance-target mean torso pitch/roll tilt threshold.")
parser.add_argument("--joint_prior_error_threshold", type=float, default=0.45, help="Training-milestone mean all-joint prior error in radians.")
parser.add_argument("--target_joint_prior_error_threshold", type=float, default=0.30, help="Acceptance-target mean all-joint prior error in radians.")
parser.add_argument("--right_arm_prior_error_threshold", type=float, default=0.35, help="Training-milestone mean right-arm joint prior error in radians.")
parser.add_argument("--target_right_arm_prior_error_threshold", type=float, default=0.25, help="Acceptance-target mean right-arm joint prior error in radians.")
parser.add_argument("--root_orientation_error_threshold_deg", type=float, default=18.0, help="Training-milestone root orientation prior error.")
parser.add_argument("--target_root_orientation_error_threshold_deg", type=float, default=12.0, help="Acceptance-target root orientation prior error.")
parser.add_argument(
    "--upright_root_on_reset",
    choices=("auto", "always", "never"),
    default="auto",
    help="Use an upright root orientation when locking deterministic priors. Auto enables this for stance tasks.",
)
parser.add_argument("--root_offset_z", type=float, default=0.0, help="Additional z offset for deterministic prior reset.")
parser.add_argument("--coverage_threshold", type=float, default=0.70, help="Training-milestone row coverage threshold.")
parser.add_argument("--target_coverage_threshold", type=float, default=0.85, help="Acceptance-target row coverage threshold.")
parser.add_argument("--backtracking_threshold", type=float, default=0.15, help="Allowed active-stroke backward-progress ratio.")
parser.add_argument("--brush_link", type=str, default="right_hand_index_1_link", help="Brush proxy link name.")
parser.add_argument(
    "--trace_prior_ids",
    type=str,
    default="",
    help="Comma-separated prior IDs for per-step active-stroke trace rows.",
)
parser.add_argument("--trace_stride", type=int, default=5, help="Record one trace row every N policy steps.")
parser.add_argument("--zero_actions", action="store_true", help="Evaluate deterministic zero residual actions instead of a checkpoint policy.")
parser.add_argument(
    "--reference_actions",
    action="store_true",
    help="Drive JointPositionAction toward the time-aligned motion reference.",
)
parser.add_argument(
    "--action_smoothing_alpha",
    type=float,
    default=1.0,
    help="Low-pass action blend factor. 1.0 disables smoothing; lower values use more previous action.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
import isaaclab.utils.math as math_utils
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.wall_brush_success import (
    select_suspicious_prior_for_visual_review,
    summarize_27_prior_metrics,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder


installed_version = metadata.version("rsl-rl-lib")


TORSO_HEAD_NAMES = [
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
]
RIGHT_HAND_NAMES = [
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
]
RIGHT_ARM_NAMES = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    *RIGHT_HAND_NAMES,
]
RIGHT_NONHAND_ARM_NAMES = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
]
LEFT_ARM_NAMES = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_hand_index_0_link",
    "left_hand_index_1_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
]
LEFT_HAND_NAMES = [
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_hand_index_0_link",
    "left_hand_index_1_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
]
LEG_NAMES = [
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
ILLEGAL_WALL_CONTACT_NAMES = TORSO_HEAD_NAMES + LEFT_ARM_NAMES + RIGHT_NONHAND_ARM_NAMES + LEG_NAMES
FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
SELF_COLLISION_PAIR_GROUPS = [
    ("right_hand_torso", RIGHT_HAND_NAMES, TORSO_HEAD_NAMES, 0.075),
    ("left_hand_torso", LEFT_HAND_NAMES, TORSO_HEAD_NAMES, 0.075),
    ("right_forearm_torso", RIGHT_NONHAND_ARM_NAMES, TORSO_HEAD_NAMES, 0.065),
    ("left_forearm_torso", ["left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link"], TORSO_HEAD_NAMES, 0.065),
    ("hands_cross", RIGHT_HAND_NAMES, LEFT_HAND_NAMES, 0.050),
    ("right_arm_left_arm", RIGHT_NONHAND_ARM_NAMES, LEFT_ARM_NAMES, 0.055),
    ("right_hand_legs", RIGHT_HAND_NAMES, LEG_NAMES, 0.045),
    ("left_hand_legs", LEFT_HAND_NAMES, LEG_NAMES, 0.045),
]


def _use_upright_root_on_reset() -> bool:
    if args_cli.upright_root_on_reset == "always":
        return True
    if args_cli.upright_root_on_reset == "never":
        return False
    return "Stance" in args_cli.task


def _build_runner(env, agent_cfg):
    if agent_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "DistillationRunner":
        return DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")


def _reset_to_prior_ids(raw_env, num_envs: int):
    """Place each env at the first frame of a deterministic prior motion."""
    device = raw_env.device
    env_ids = torch.arange(num_envs, device=device)
    prior_ids = torch.arange(num_envs, device=device, dtype=torch.long) % int(raw_env.total_motions)
    raw_env.motion_ids[env_ids] = prior_ids
    raw_env.start_motion_times[env_ids] = 0.0
    raw_env.episode_length_buf[env_ids] = 0

    motion_res = raw_env.motion_lib.get_motion_state(prior_ids, torch.zeros(num_envs, device=device))
    asset: Articulation = raw_env.scene["robot"]
    joint_ids = torch.tensor([asset.joint_names.index(name) for name in JointNamesOrder], device=device)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos[:, joint_ids] = motion_res["dof_pos"]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    positions = motion_res["root_pos"] + raw_env.scene.env_origins[env_ids]
    positions[:, 2] += float(args_cli.root_offset_z)
    if _use_upright_root_on_reset():
        orientations = torch.zeros((num_envs, 4), device=device)
        orientations[:, 0] = 1.0
    else:
        orientations = motion_res["root_rot"]
    velocities = torch.zeros((num_envs, 6), device=device)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    raw_env.scene.write_data_to_sim()
    raw_env.sim.forward()
    raw_env.scene.update(raw_env.step_dt)
    return prior_ids.clone()


def _body_ids(asset: Articulation, names: list[str], device) -> torch.Tensor:
    ids = [asset.body_names.index(name) for name in names if name in asset.body_names]
    if not ids:
        raise ValueError(f"None of the requested body names exist: {names}")
    return torch.tensor(ids, device=device, dtype=torch.long)


def _optional_body_ids(asset: Articulation, names: list[str], device) -> torch.Tensor:
    ids = [asset.body_names.index(name) for name in names if name in asset.body_names]
    return torch.tensor(ids, device=device, dtype=torch.long)


def _joint_order_indices(names: list[str], device) -> torch.Tensor:
    ids = [JointNamesOrder.index(name) for name in names if name in JointNamesOrder]
    if not ids:
        raise ValueError(f"None of the requested joint names exist in JointNamesOrder: {names}")
    return torch.tensor(ids, device=device, dtype=torch.long)


def _group_min_clearance(body_clearance: torch.Tensor, body_ids: torch.Tensor) -> torch.Tensor:
    return torch.min(body_clearance[:, body_ids], dim=1).values


def _group_positions(body_pos: torch.Tensor, body_ids: torch.Tensor) -> torch.Tensor:
    return body_pos[:, body_ids, :]


def _wall_contact_sensor(raw_env):
    try:
        return raw_env.scene.sensors["contact_forces"]
    except KeyError:
        return None


def _wall_contact_body_pairs(raw_env, asset: Articulation, names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = _wall_contact_sensor(raw_env)
    if sensor is None:
        empty = torch.empty(0, device=raw_env.device, dtype=torch.long)
        return empty, empty
    wanted = set(names)
    pairs = [(idx, asset.body_names.index(name)) for idx, name in enumerate(sensor.body_names) if name in wanted and name in asset.body_names]
    if not pairs:
        empty = torch.empty(0, device=raw_env.device, dtype=torch.long)
        return empty, empty
    sensor_ids, asset_ids = zip(*pairs)
    return (
        torch.tensor(sensor_ids, device=raw_env.device, dtype=torch.long),
        torch.tensor(asset_ids, device=raw_env.device, dtype=torch.long),
    )


def _all_illegal_wall_contact_body_pairs(raw_env, asset: Articulation, brush_link: str) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = _wall_contact_sensor(raw_env)
    if sensor is None:
        empty = torch.empty(0, device=raw_env.device, dtype=torch.long)
        return empty, empty
    pairs = [
        (idx, asset.body_names.index(name))
        for idx, name in enumerate(sensor.body_names)
        if name != brush_link and name in asset.body_names
    ]
    if not pairs:
        empty = torch.empty(0, device=raw_env.device, dtype=torch.long)
        return empty, empty
    sensor_ids, asset_ids = zip(*pairs)
    return (
        torch.tensor(sensor_ids, device=raw_env.device, dtype=torch.long),
        torch.tensor(asset_ids, device=raw_env.device, dtype=torch.long),
    )


def _wall_contact_forces(
    raw_env,
    sensor_body_ids: torch.Tensor,
    asset_body_ids: torch.Tensor,
    body_clearance: torch.Tensor,
) -> torch.Tensor:
    if sensor_body_ids.numel() == 0:
        return torch.zeros((raw_env.num_envs, 0), device=raw_env.device)
    sensor = _wall_contact_sensor(raw_env)
    if sensor is None or sensor.data.net_forces_w is None:
        return torch.zeros((raw_env.num_envs, int(sensor_body_ids.numel())), device=raw_env.device)
    wall_normal_force = torch.abs(sensor.data.net_forces_w[:, sensor_body_ids, 0])
    near_wall = body_clearance[:, asset_body_ids] < args_cli.wall_contact_near_margin
    return torch.where(near_wall, wall_normal_force, torch.zeros_like(wall_normal_force))


def _max_wall_contact_force(
    raw_env,
    sensor_body_ids: torch.Tensor,
    asset_body_ids: torch.Tensor,
    body_clearance: torch.Tensor,
) -> torch.Tensor:
    forces = _wall_contact_forces(raw_env, sensor_body_ids, asset_body_ids, body_clearance)
    if forces.shape[1] == 0:
        return torch.zeros(raw_env.num_envs, device=raw_env.device)
    return forces.amax(dim=1)


def _termination_term(raw_env, name: str) -> torch.Tensor | None:
    manager = getattr(raw_env, "termination_manager", None)
    if manager is None or name not in manager.active_terms:
        return None
    return manager.get_term(name)


def _self_collision_proxy_stats(raw_env, asset: Articulation, body_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    violations = []
    margins = []
    for _, group_a, group_b, min_distance in SELF_COLLISION_PAIR_GROUPS:
        ids_a = _optional_body_ids(asset, group_a, raw_env.device)
        ids_b = _optional_body_ids(asset, group_b, raw_env.device)
        if ids_a.numel() == 0 or ids_b.numel() == 0:
            continue
        delta = body_pos[:, ids_a, None, :] - body_pos[:, None, ids_b, :]
        pair_dist = torch.norm(delta, dim=-1).reshape(raw_env.num_envs, -1)
        min_dist = torch.min(pair_dist, dim=1).values
        margins.append(min_dist - min_distance)
        violations.append(((min_distance - min_dist).clamp_min(0.0) / 0.04) ** 2)
    if not violations:
        empty = torch.zeros(raw_env.num_envs, device=raw_env.device)
        return empty, empty
    return torch.max(torch.stack(violations, dim=1), dim=1).values, torch.min(torch.stack(margins, dim=1), dim=1).values


def _self_collision_proxy_group_stats(raw_env, asset: Articulation, body_pos: torch.Tensor):
    stats = {}
    for name, group_a, group_b, min_distance in SELF_COLLISION_PAIR_GROUPS:
        ids_a = _optional_body_ids(asset, group_a, raw_env.device)
        ids_b = _optional_body_ids(asset, group_b, raw_env.device)
        if ids_a.numel() == 0 or ids_b.numel() == 0:
            continue
        delta = body_pos[:, ids_a, None, :] - body_pos[:, None, ids_b, :]
        pair_dist = torch.norm(delta, dim=-1).reshape(raw_env.num_envs, -1)
        min_dist = torch.min(pair_dist, dim=1).values
        margin = min_dist - min_distance
        violation = ((min_distance - min_dist).clamp_min(0.0) / 0.04) ** 2
        stats[name] = (violation, margin)
    return stats


def _torso_yaw_and_upright(asset: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    root_quat = math_utils.quat_unique(asset.data.root_quat_w)
    root_rot = math_utils.matrix_from_quat(root_quat)
    forward = root_rot[:, :, 0]
    forward_xy = forward[:, :2]
    forward_xy = forward_xy / torch.norm(forward_xy, dim=1, keepdim=True).clamp_min(1e-6)
    wall_normal_xy = torch.zeros_like(forward_xy)
    wall_normal_xy[:, 0] = 1.0
    yaw_cos = torch.sum(forward_xy * wall_normal_xy, dim=1).clamp(-1.0, 1.0)
    yaw = torch.rad2deg(torch.acos(yaw_cos))

    z_axis = root_rot[:, :, 2]
    upright = torch.rad2deg(torch.acos(z_axis[:, 2].clamp(-1.0, 1.0)))
    return yaw, upright


def _measure(raw_env, brush_link: str):
    device = raw_env.device
    motion_times = raw_env.episode_length_buf * raw_env.step_dt + raw_env.start_motion_times.to(
        device=device, dtype=torch.float32
    )
    motion_res = raw_env.motion_lib.get_motion_state(raw_env.motion_ids, motion_times)
    asset: Articulation = raw_env.scene["robot"]
    body_id = asset.body_names.index(brush_link)
    brush_tip = asset.data.body_state_w[:, body_id, :3] - raw_env.scene.env_origins

    wall_x_error = torch.abs(brush_tip[:, 0] - motion_res["wall_mid"][:, 0])
    row_yz_error = torch.norm(brush_tip[:, 1:3] - motion_res["brush_tip_pos"][:, 1:3], dim=1)
    reference_error = torch.norm(brush_tip - motion_res["brush_tip_pos"], dim=1)

    start = motion_res["wall_start"]
    end = motion_res["wall_end"]
    line = end - start
    denom = torch.sum(line * line, dim=1).clamp_min(1e-6)
    tip_phase = torch.sum((brush_tip - start) * line, dim=1) / denom
    ref_phase = torch.sum((motion_res["brush_tip_pos"] - start) * line, dim=1) / denom
    phase_error = torch.abs(torch.clamp(tip_phase, 0.0, 1.0) - torch.clamp(ref_phase, 0.0, 1.0))
    body_pos = asset.data.body_state_w[:, :, :3] - raw_env.scene.env_origins.unsqueeze(1)
    wall_x = motion_res["wall_mid"][:, 0]
    body_clearance = wall_x.unsqueeze(1) - body_pos[:, :, 0]
    torso_yaw_deg, torso_upright_deg = _torso_yaw_and_upright(asset)
    joint_ids = torch.tensor([asset.joint_names.index(name) for name in JointNamesOrder], device=device)
    joint_error = torch.abs(asset.data.joint_pos[:, joint_ids] - motion_res["dof_pos"])
    right_arm_joint_error = joint_error[:, _joint_order_indices(RIGHT_ARM_JOINT_NAMES, device)]
    root_pos = asset.data.root_pos_w - raw_env.scene.env_origins
    root_position_error = torch.norm(root_pos - motion_res["root_pos"], dim=1)
    root_orientation_error_deg = torch.rad2deg(math_utils.quat_error_magnitude(motion_res["root_rot"], asset.data.root_quat_w))
    self_collision_violation, self_collision_margin = _self_collision_proxy_stats(raw_env, asset, body_pos)

    return {
        "active": motion_res["stroke_active"].bool(),
        "brush_tip": brush_tip,
        "wall_x_error": wall_x_error,
        "row_yz_error": row_yz_error,
        "reference_error": reference_error,
        "phase_error": phase_error,
        "progress": torch.clamp(tip_phase, 0.0, 1.0),
        "wall_start": motion_res["wall_start"],
        "wall_mid": motion_res["wall_mid"],
        "wall_end": motion_res["wall_end"],
        "body_pos": body_pos,
        "body_clearance": body_clearance,
        "torso_yaw_deg": torso_yaw_deg,
        "torso_upright_deg": torso_upright_deg,
        "mean_joint_prior_error_rad": torch.mean(joint_error, dim=1),
        "mean_right_arm_joint_prior_error_rad": torch.mean(right_arm_joint_error, dim=1),
        "root_position_error_m": root_position_error,
        "root_orientation_error_deg": root_orientation_error_deg,
        "self_collision_proxy_violation": self_collision_violation,
        "self_collision_margin_m": self_collision_margin,
    }


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def _policy_obs(env):
    obs_result = env.get_observations()
    return obs_result[0] if isinstance(obs_result, tuple) else obs_result


def action_mode_reference(raw_env) -> torch.Tensor:
    action_term = raw_env.action_manager.get_term("joint_pos")
    motion_times = raw_env.episode_length_buf * raw_env.step_dt + raw_env.start_motion_times.to(
        device=raw_env.device, dtype=torch.float32
    )
    motion_res = raw_env.motion_lib.get_motion_state(raw_env.motion_ids, motion_times)
    ref_joint_pos = motion_res["dof_pos"]
    offset = action_term._offset
    scale = action_term._scale
    if not torch.is_tensor(offset):
        offset = torch.full_like(ref_joint_pos, float(offset))
    if not torch.is_tensor(scale):
        scale = torch.full_like(ref_joint_pos, float(scale))
    return (ref_joint_pos - offset) / scale.clamp_min(1e-6)


def smooth_actions(actions: torch.Tensor, previous_actions, alpha: float) -> torch.Tensor:
    alpha = max(0.0, min(1.0, float(alpha)))
    if previous_actions is None or alpha >= 1.0:
        return actions
    return alpha * actions + (1.0 - alpha) * previous_actions


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg.observations, "policy") and hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise FileNotFoundError(f"No published pretrained checkpoint for {train_task_name}")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.zero_actions or args_cli.reference_actions:
        runner = None
        policy = None
    else:
        runner = _build_runner(env, agent_cfg)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    prior_ids = _reset_to_prior_ids(raw, args_cli.num_envs)
    obs = _policy_obs(env)
    trace_prior_ids = {int(value) for value in args_cli.trace_prior_ids.split(",") if value.strip()}
    trace_stride = max(1, int(args_cli.trace_stride))
    trace_rows = []
    smoothing_alpha = max(0.0, min(1.0, float(args_cli.action_smoothing_alpha)))
    smoothed_actions = None

    device = raw.device
    num_envs = args_cli.num_envs
    action_dim = int(raw.action_manager.total_action_dim)
    right_arm_action_ids = _joint_order_indices(RIGHT_ARM_JOINT_NAMES, device)
    right_arm_action_ids = right_arm_action_ids[right_arm_action_ids < action_dim]
    asset: Articulation = raw.scene["robot"]
    all_body_ids = torch.arange(len(asset.body_names), device=device, dtype=torch.long)
    illegal_body_ids = _body_ids(asset, ILLEGAL_WALL_CONTACT_NAMES, device)
    torso_head_ids = _body_ids(asset, TORSO_HEAD_NAMES, device)
    right_arm_ids = _body_ids(asset, RIGHT_ARM_NAMES, device)
    right_hand_ids = _body_ids(asset, RIGHT_HAND_NAMES, device)
    left_arm_ids = _body_ids(asset, LEFT_ARM_NAMES, device)
    foot_ids = _body_ids(asset, FOOT_NAMES, device)
    illegal_wall_sensor_ids, illegal_wall_asset_ids = _wall_contact_body_pairs(raw, asset, ILLEGAL_WALL_CONTACT_NAMES)
    torso_wall_sensor_ids, torso_wall_asset_ids = _wall_contact_body_pairs(raw, asset, TORSO_HEAD_NAMES)
    right_hand_wall_sensor_ids = torch.empty(0, device=device, dtype=torch.long)
    right_hand_wall_asset_ids = torch.empty(0, device=device, dtype=torch.long)
    left_arm_wall_sensor_ids, left_arm_wall_asset_ids = _wall_contact_body_pairs(raw, asset, LEFT_ARM_NAMES)
    initial_body_pos = asset.data.body_state_w[:, :, :3] - raw.scene.env_origins.unsqueeze(1)
    initial_foot_xy = initial_body_pos[:, foot_ids, :2].clone()

    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    done_step = torch.full((num_envs,), args_cli.num_steps, dtype=torch.int32, device=device)
    non_timeout_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_active_step = torch.full((num_envs,), -1, dtype=torch.int32, device=device)
    eval_steps = torch.zeros(num_envs, device=device)
    active_steps = torch.zeros(num_envs, device=device)
    contact_hits = torch.zeros(num_envs, device=device)
    target_contact_hits = torch.zeros(num_envs, device=device)
    row_hits = torch.zeros(num_envs, device=device)
    target_row_hits = torch.zeros(num_envs, device=device)
    combined_hits = torch.zeros(num_envs, device=device)
    target_combined_hits = torch.zeros(num_envs, device=device)
    ordered_anchor_hits = torch.zeros(num_envs, device=device)
    target_ordered_anchor_hits = torch.zeros(num_envs, device=device)
    next_anchor_index = torch.zeros(num_envs, device=device, dtype=torch.long)
    target_next_anchor_index = torch.zeros(num_envs, device=device, dtype=torch.long)
    illegal_hits = torch.zeros(num_envs, device=device)
    torso_illegal_hits = torch.zeros(num_envs, device=device)
    right_hand_illegal_hits = torch.zeros(num_envs, device=device)
    left_arm_illegal_hits = torch.zeros(num_envs, device=device)
    wall_contact_termination_hits = torch.zeros(num_envs, device=device)
    self_collision_termination_hits = torch.zeros(num_envs, device=device)
    torso_below_termination_hits = torch.zeros(num_envs, device=device)
    torso_angle_termination_hits = torch.zeros(num_envs, device=device)
    min_root_z = torch.full((num_envs,), float("inf"), device=device)
    min_root_cos_z = torch.full((num_envs,), float("inf"), device=device)
    sum_wall_x_error = torch.zeros(num_envs, device=device)
    sum_row_yz_error = torch.zeros(num_envs, device=device)
    sum_reference_error = torch.zeros(num_envs, device=device)
    sum_phase_error = torch.zeros(num_envs, device=device)
    sum_torso_yaw = torch.zeros(num_envs, device=device)
    sum_torso_upright = torch.zeros(num_envs, device=device)
    sum_joint_prior_error = torch.zeros(num_envs, device=device)
    sum_right_arm_joint_prior_error = torch.zeros(num_envs, device=device)
    sum_root_position_error = torch.zeros(num_envs, device=device)
    sum_root_orientation_error = torch.zeros(num_envs, device=device)
    max_progress = torch.zeros(num_envs, device=device)
    min_progress = torch.ones(num_envs, device=device)
    env_buffer_active_steps = torch.zeros(num_envs, device=device)
    env_buffer_contact_ratio = torch.zeros(num_envs, device=device)
    env_buffer_row_ratio = torch.zeros(num_envs, device=device)
    env_buffer_combined_ratio = torch.zeros(num_envs, device=device)
    env_buffer_coverage = torch.zeros(num_envs, device=device)
    env_buffer_next_anchor = torch.zeros(num_envs, device=device)
    env_buffer_n_successes = torch.zeros(num_envs, device=device)
    prev_progress = torch.zeros(num_envs, device=device)
    has_prev_progress = torch.zeros(num_envs, dtype=torch.bool, device=device)
    forward_progress = torch.zeros(num_envs, device=device)
    backward_progress = torch.zeros(num_envs, device=device)
    min_torso_clearance = torch.full((num_envs,), float("inf"), device=device)
    min_nonbrush_clearance = torch.full((num_envs,), float("inf"), device=device)
    min_right_hand_clearance = torch.full((num_envs,), float("inf"), device=device)
    min_left_arm_clearance = torch.full((num_envs,), float("inf"), device=device)
    max_illegal_wall_contact_force = torch.zeros(num_envs, device=device)
    max_self_collision_proxy_violation = torch.zeros(num_envs, device=device)
    min_self_collision_margin = torch.full((num_envs,), float("inf"), device=device)
    self_collision_group_max = {
        name: torch.zeros(num_envs, device=device) for name, *_ in SELF_COLLISION_PAIR_GROUPS
    }
    self_collision_group_min_margin = {
        name: torch.full((num_envs,), float("inf"), device=device) for name, *_ in SELF_COLLISION_PAIR_GROUPS
    }
    max_foot_slip = torch.zeros(num_envs, device=device)
    prev_actions = torch.zeros((num_envs, action_dim), device=device)
    has_prev_actions = torch.zeros(num_envs, dtype=torch.bool, device=device)
    active_action_delta_steps = torch.zeros(num_envs, device=device)
    sum_active_action_delta_l2 = torch.zeros(num_envs, device=device)
    max_active_action_delta_l2 = torch.zeros(num_envs, device=device)
    sum_active_right_arm_action_delta_l2 = torch.zeros(num_envs, device=device)
    max_active_right_arm_action_delta_l2 = torch.zeros(num_envs, device=device)
    prev_brush_tip = torch.zeros((num_envs, 3), device=device)
    has_prev_brush_tip = torch.zeros(num_envs, dtype=torch.bool, device=device)
    prev_brush_tip_vel = torch.zeros((num_envs, 3), device=device)
    has_prev_brush_tip_vel = torch.zeros(num_envs, dtype=torch.bool, device=device)
    prev_brush_tip_accel = torch.zeros((num_envs, 3), device=device)
    has_prev_brush_tip_accel = torch.zeros(num_envs, dtype=torch.bool, device=device)
    brush_tip_vel_steps = torch.zeros(num_envs, device=device)
    brush_tip_accel_steps = torch.zeros(num_envs, device=device)
    brush_tip_jerk_steps = torch.zeros(num_envs, device=device)
    sum_brush_tip_speed_mps = torch.zeros(num_envs, device=device)
    max_brush_tip_speed_mps = torch.zeros(num_envs, device=device)
    sum_brush_tip_accel_mps2 = torch.zeros(num_envs, device=device)
    max_brush_tip_accel_mps2 = torch.zeros(num_envs, device=device)
    sum_brush_tip_jerk_mps3 = torch.zeros(num_envs, device=device)
    max_brush_tip_jerk_mps3 = torch.zeros(num_envs, device=device)
    row_error_samples = [[] for _ in range(num_envs)]
    torso_yaw_samples = [[] for _ in range(num_envs)]

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            alive_before_step = alive.clone()
            if hasattr(raw, "wall_brush_active_steps"):
                buf_active = raw.wall_brush_active_steps.clamp_min(1.0)
                env_buffer_active_steps = torch.maximum(
                    env_buffer_active_steps,
                    torch.where(alive_before_step, raw.wall_brush_active_steps, env_buffer_active_steps),
                )
                env_buffer_contact_ratio = torch.maximum(
                    env_buffer_contact_ratio,
                    torch.where(
                        alive_before_step,
                        raw.wall_brush_contact_steps / buf_active,
                        env_buffer_contact_ratio,
                    ),
                )
                env_buffer_row_ratio = torch.maximum(
                    env_buffer_row_ratio,
                    torch.where(alive_before_step, raw.wall_brush_row_steps / buf_active, env_buffer_row_ratio),
                )
                env_buffer_combined_ratio = torch.maximum(
                    env_buffer_combined_ratio,
                    torch.where(
                        alive_before_step,
                        raw.wall_brush_combined_steps / buf_active,
                        env_buffer_combined_ratio,
                    ),
                )
                finite_phase = torch.isfinite(raw.wall_brush_min_phase) & torch.isfinite(raw.wall_brush_max_phase)
                buf_coverage = torch.where(
                    finite_phase,
                    (raw.wall_brush_max_phase - raw.wall_brush_min_phase).clamp_min(0.0),
                    torch.zeros_like(raw.wall_brush_active_steps),
                )
                env_buffer_coverage = torch.maximum(
                    env_buffer_coverage,
                    torch.where(alive_before_step, buf_coverage, env_buffer_coverage),
                )
                env_buffer_next_anchor = torch.maximum(
                    env_buffer_next_anchor,
                    torch.where(alive_before_step, raw.wall_brush_next_anchor.float(), env_buffer_next_anchor),
                )
            if hasattr(raw, "n_successes"):
                env_buffer_n_successes = torch.maximum(
                    env_buffer_n_successes,
                    torch.where(alive_before_step, raw.n_successes.float(), env_buffer_n_successes),
                )
            eval_steps += alive_before_step.float()
            if args_cli.reference_actions:
                actions = action_mode_reference(raw)
            elif args_cli.zero_actions:
                actions = torch.zeros((num_envs, action_dim), device=device)
            else:
                actions = policy(obs)
            actions = smooth_actions(actions, smoothed_actions, smoothing_alpha)
            smoothed_actions = actions.clone()
            action_delta = torch.norm(actions - prev_actions, dim=1)
            if right_arm_action_ids.numel() > 0:
                right_arm_action_delta = torch.norm(
                    actions[:, right_arm_action_ids] - prev_actions[:, right_arm_action_ids], dim=1
                )
            else:
                right_arm_action_delta = torch.zeros(num_envs, device=device)
            obs, _, dones, _ = env.step(actions)
            if policy is not None and version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            if smoothed_actions is not None:
                smoothed_actions[dones.bool()] = 0.0

            root_pos_env = asset.data.root_pos_w - raw.scene.env_origins
            root_rot = math_utils.matrix_from_quat(math_utils.quat_unique(asset.data.root_quat_w))
            root_cos_z = root_rot[:, 2, 2]
            min_root_z = torch.minimum(min_root_z, torch.where(alive_before_step, root_pos_env[:, 2], min_root_z))
            min_root_cos_z = torch.minimum(
                min_root_cos_z, torch.where(alive_before_step, root_cos_z, min_root_cos_z)
            )

            dones_bool = dones.bool()
            torso_below_done = _termination_term(raw, "torso_below_threshold")
            if torso_below_done is not None:
                torso_below_termination_hits += (torso_below_done.bool() & alive_before_step).float()
            torso_angle_done = _termination_term(raw, "torso_angle_below_threshold")
            if torso_angle_done is not None:
                torso_angle_termination_hits += (torso_angle_done.bool() & alive_before_step).float()
            wall_contact_done = _termination_term(raw, "nonbrush_wall_contact")
            if wall_contact_done is not None:
                wall_contact_done = wall_contact_done & alive_before_step
                wall_contact_termination_hits += wall_contact_done.float()
                illegal_hits += wall_contact_done.float()
            self_collision_done = _termination_term(raw, "self_collision_proxy")
            if self_collision_done is not None:
                self_collision_done = self_collision_done & alive_before_step
                self_collision_termination_hits += self_collision_done.float()
            timeout_done = _termination_term(raw, "time_out")
            if timeout_done is None:
                timeout_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
            timeout_done = timeout_done.bool() & alive_before_step
            newly_done = dones_bool & alive
            done_step[newly_done] = step + 1
            non_timeout_done |= newly_done & (~timeout_done)
            alive = alive & ~dones_bool
            if not torch.any(alive):
                break

            metrics = _measure(raw, args_cli.brush_link)
            active = metrics["active"] & alive
            active_f = active.float()
            active_steps += active_f
            newly_active = active & (first_active_step < 0)
            first_active_step[newly_active] = step + 1

            action_delta_mask = active & has_prev_actions
            active_action_delta_steps += action_delta_mask.float()
            sum_active_action_delta_l2 += action_delta * action_delta_mask.float()
            max_active_action_delta_l2 = torch.maximum(
                max_active_action_delta_l2, torch.where(action_delta_mask, action_delta, max_active_action_delta_l2)
            )
            sum_active_right_arm_action_delta_l2 += right_arm_action_delta * action_delta_mask.float()
            max_active_right_arm_action_delta_l2 = torch.maximum(
                max_active_right_arm_action_delta_l2,
                torch.where(action_delta_mask, right_arm_action_delta, max_active_right_arm_action_delta_l2),
            )

            dt = float(raw.step_dt)
            tip_vel = (metrics["brush_tip"] - prev_brush_tip) / dt
            tip_vel_mask = active & has_prev_brush_tip
            tip_speed = torch.norm(tip_vel, dim=1)
            brush_tip_vel_steps += tip_vel_mask.float()
            sum_brush_tip_speed_mps += tip_speed * tip_vel_mask.float()
            max_brush_tip_speed_mps = torch.maximum(
                max_brush_tip_speed_mps, torch.where(tip_vel_mask, tip_speed, max_brush_tip_speed_mps)
            )

            tip_accel = (tip_vel - prev_brush_tip_vel) / dt
            tip_accel_mask = active & has_prev_brush_tip_vel
            tip_accel_norm = torch.norm(tip_accel, dim=1)
            brush_tip_accel_steps += tip_accel_mask.float()
            sum_brush_tip_accel_mps2 += tip_accel_norm * tip_accel_mask.float()
            max_brush_tip_accel_mps2 = torch.maximum(
                max_brush_tip_accel_mps2,
                torch.where(tip_accel_mask, tip_accel_norm, max_brush_tip_accel_mps2),
            )

            tip_jerk = (tip_accel - prev_brush_tip_accel) / dt
            tip_jerk_mask = active & has_prev_brush_tip_accel
            tip_jerk_norm = torch.norm(tip_jerk, dim=1)
            brush_tip_jerk_steps += tip_jerk_mask.float()
            sum_brush_tip_jerk_mps3 += tip_jerk_norm * tip_jerk_mask.float()
            max_brush_tip_jerk_mps3 = torch.maximum(
                max_brush_tip_jerk_mps3,
                torch.where(tip_jerk_mask, tip_jerk_norm, max_brush_tip_jerk_mps3),
            )

            contact_ok = (metrics["wall_x_error"] <= args_cli.contact_x_threshold) & active
            row_ok = (metrics["row_yz_error"] <= args_cli.row_yz_threshold) & active
            target_row_ok = (metrics["row_yz_error"] <= args_cli.target_row_yz_threshold) & active
            contact_hits += contact_ok.float()
            target_contact_hits += contact_ok.float()
            row_hits += row_ok.float()
            target_row_hits += target_row_ok.float()
            combined_hits += (contact_ok & row_ok).float()
            target_combined_hits += (contact_ok & target_row_ok).float()

            anchors = torch.stack([metrics["wall_start"], metrics["wall_mid"], metrics["wall_end"]], dim=1)
            expected_anchor = anchors[torch.arange(num_envs, device=device), next_anchor_index.clamp(max=2)]
            anchor_dist = torch.norm(metrics["brush_tip"][:, 1:3] - expected_anchor[:, 1:3], dim=1)
            anchor_hit = active & contact_ok & (next_anchor_index < 3) & (anchor_dist <= args_cli.row_yz_threshold)
            next_anchor_index[anchor_hit] += 1
            ordered_anchor_hits += anchor_hit.float()

            target_expected_anchor = anchors[torch.arange(num_envs, device=device), target_next_anchor_index.clamp(max=2)]
            target_anchor_dist = torch.norm(metrics["brush_tip"][:, 1:3] - target_expected_anchor[:, 1:3], dim=1)
            target_anchor_hit = (
                active
                & contact_ok
                & (target_next_anchor_index < 3)
                & (target_anchor_dist <= args_cli.target_row_yz_threshold)
            )
            target_next_anchor_index[target_anchor_hit] += 1
            target_ordered_anchor_hits += target_anchor_hit.float()

            sum_wall_x_error += metrics["wall_x_error"] * active_f
            sum_row_yz_error += metrics["row_yz_error"] * active_f
            sum_reference_error += metrics["reference_error"] * active_f
            sum_phase_error += metrics["phase_error"] * active_f
            sum_torso_yaw += metrics["torso_yaw_deg"] * active_f
            sum_torso_upright += metrics["torso_upright_deg"] * active_f
            sum_joint_prior_error += metrics["mean_joint_prior_error_rad"] * active_f
            sum_right_arm_joint_prior_error += metrics["mean_right_arm_joint_prior_error_rad"] * active_f
            sum_root_position_error += metrics["root_position_error_m"] * active_f
            sum_root_orientation_error += metrics["root_orientation_error_deg"] * active_f

            torso_clearance = _group_min_clearance(metrics["body_clearance"], torso_head_ids)
            nonbrush_clearance = _group_min_clearance(metrics["body_clearance"], illegal_body_ids)
            right_hand_clearance = _group_min_clearance(metrics["body_clearance"], right_hand_ids)
            left_arm_clearance = _group_min_clearance(metrics["body_clearance"], left_arm_ids)
            min_torso_clearance = torch.minimum(min_torso_clearance, torch.where(active, torso_clearance, min_torso_clearance))
            min_nonbrush_clearance = torch.minimum(
                min_nonbrush_clearance, torch.where(active, nonbrush_clearance, min_nonbrush_clearance)
            )
            min_right_hand_clearance = torch.minimum(
                min_right_hand_clearance, torch.where(active, right_hand_clearance, min_right_hand_clearance)
            )
            min_left_arm_clearance = torch.minimum(
                min_left_arm_clearance, torch.where(active, left_arm_clearance, min_left_arm_clearance)
            )
            illegal_wall_force = _max_wall_contact_force(
                raw, illegal_wall_sensor_ids, illegal_wall_asset_ids, metrics["body_clearance"]
            )
            torso_wall_force = _max_wall_contact_force(raw, torso_wall_sensor_ids, torso_wall_asset_ids, metrics["body_clearance"])
            right_hand_wall_force = _max_wall_contact_force(
                raw, right_hand_wall_sensor_ids, right_hand_wall_asset_ids, metrics["body_clearance"]
            )
            left_arm_wall_force = _max_wall_contact_force(
                raw, left_arm_wall_sensor_ids, left_arm_wall_asset_ids, metrics["body_clearance"]
            )
            max_illegal_wall_contact_force = torch.maximum(
                max_illegal_wall_contact_force, torch.where(alive, illegal_wall_force, max_illegal_wall_contact_force)
            )
            max_self_collision_proxy_violation = torch.maximum(
                max_self_collision_proxy_violation,
                torch.where(alive, metrics["self_collision_proxy_violation"], max_self_collision_proxy_violation),
            )
            min_self_collision_margin = torch.minimum(
                min_self_collision_margin,
                torch.where(alive, metrics["self_collision_margin_m"], min_self_collision_margin),
            )
            for group_name, (group_violation, group_margin) in _self_collision_proxy_group_stats(
                raw, asset, metrics["body_pos"]
            ).items():
                self_collision_group_max[group_name] = torch.maximum(
                    self_collision_group_max[group_name],
                    torch.where(alive, group_violation, self_collision_group_max[group_name]),
                )
                self_collision_group_min_margin[group_name] = torch.minimum(
                    self_collision_group_min_margin[group_name],
                    torch.where(alive, group_margin, self_collision_group_min_margin[group_name]),
                )
            torso_illegal = (torso_wall_force > args_cli.wall_contact_force_threshold) & alive
            right_hand_illegal = (right_hand_wall_force > args_cli.wall_contact_force_threshold) & alive
            left_arm_illegal = (left_arm_wall_force > args_cli.wall_contact_force_threshold) & alive
            nonbrush_illegal = (illegal_wall_force > args_cli.wall_contact_force_threshold) & alive
            torso_illegal_hits += torso_illegal.float()
            right_hand_illegal_hits += right_hand_illegal.float()
            left_arm_illegal_hits += left_arm_illegal.float()
            illegal_hits += (torso_illegal | right_hand_illegal | left_arm_illegal | nonbrush_illegal).float()

            foot_xy = _group_positions(metrics["body_pos"], foot_ids)[:, :, :2]
            foot_slip = torch.norm(foot_xy - initial_foot_xy, dim=2).max(dim=1).values
            max_foot_slip = torch.maximum(max_foot_slip, torch.where(active, foot_slip, max_foot_slip))

            progress = metrics["progress"]
            max_progress = torch.maximum(max_progress, torch.where(active, progress, max_progress))
            min_progress = torch.minimum(min_progress, torch.where(active, progress, min_progress))
            prev_active = active & has_prev_progress
            delta = progress - prev_progress
            forward_progress += torch.where(prev_active, torch.clamp(delta, min=0.0), torch.zeros_like(delta))
            backward_progress += torch.where(prev_active, torch.clamp(-delta, min=0.0), torch.zeros_like(delta))
            prev_progress = torch.where(active, progress, prev_progress)
            has_prev_progress |= active

            active_ids = active.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            if trace_prior_ids and step % trace_stride == 0:
                for env_id in active_ids:
                    prior_id = int(prior_ids[env_id].item())
                    if prior_id not in trace_prior_ids:
                        continue
                    trace_rows.append(
                        {
                            "step": step + 1,
                            "env_id": int(env_id),
                            "prior_id": prior_id,
                            "progress": float(progress[env_id].item()),
                            "next_anchor_index": int(next_anchor_index[env_id].item()),
                            "target_next_anchor_index": int(target_next_anchor_index[env_id].item()),
                            "anchor_dist_m": float(anchor_dist[env_id].item()),
                            "target_anchor_dist_m": float(target_anchor_dist[env_id].item()),
                            "wall_x_error_m": float(metrics["wall_x_error"][env_id].item()),
                            "row_yz_error_m": float(metrics["row_yz_error"][env_id].item()),
                            "reference_error_m": float(metrics["reference_error"][env_id].item()),
                            "phase_error": float(metrics["phase_error"][env_id].item()),
                            "contact_ok": bool(contact_ok[env_id].item()),
                            "row_ok": bool(row_ok[env_id].item()),
                            "target_row_ok": bool(target_row_ok[env_id].item()),
                            "foot_slip_m": float(foot_slip[env_id].item()),
                            "action_delta_l2": float(action_delta[env_id].item()),
                            "right_arm_action_delta_l2": float(right_arm_action_delta[env_id].item()),
                            "brush_tip_speed_mps": float(tip_speed[env_id].item()),
                            "brush_tip_accel_mps2": float(tip_accel_norm[env_id].item()),
                            "brush_tip_jerk_mps3": float(tip_jerk_norm[env_id].item()),
                            "root_orientation_error_deg": float(metrics["root_orientation_error_deg"][env_id].item()),
                            "right_arm_joint_prior_error_rad": float(
                                metrics["mean_right_arm_joint_prior_error_rad"][env_id].item()
                            ),
                            "torso_yaw_deg": float(metrics["torso_yaw_deg"][env_id].item()),
                            "torso_upright_deg": float(metrics["torso_upright_deg"][env_id].item()),
                        }
                    )
            row_values = metrics["row_yz_error"].detach().cpu().tolist()
            yaw_values = metrics["torso_yaw_deg"].detach().cpu().tolist()
            for env_id in active_ids:
                row_error_samples[env_id].append(float(row_values[env_id]))
                torso_yaw_samples[env_id].append(float(yaw_values[env_id]))

            prev_actions[alive_before_step] = actions[alive_before_step]
            has_prev_actions[alive_before_step] = True
            prev_brush_tip[active] = metrics["brush_tip"][active]
            has_prev_brush_tip[active] = True
            prev_brush_tip_vel[tip_vel_mask] = tip_vel[tip_vel_mask]
            has_prev_brush_tip_vel[tip_vel_mask] = True
            prev_brush_tip_accel[tip_accel_mask] = tip_accel[tip_accel_mask]
            has_prev_brush_tip_accel[tip_accel_mask] = True

    denom = active_steps.clamp_min(1.0)
    illegal_denom = eval_steps.clamp_min(1.0)
    per_prior = []
    for idx in range(num_envs):
        active_count = float(active_steps[idx].item())
        row_p95 = 0.0
        yaw_p95 = 0.0
        if row_error_samples[idx]:
            row_sorted = sorted(row_error_samples[idx])
            row_p95 = row_sorted[min(len(row_sorted) - 1, int(0.95 * (len(row_sorted) - 1)))]
            yaw_sorted = sorted(torso_yaw_samples[idx])
            yaw_p95 = yaw_sorted[min(len(yaw_sorted) - 1, int(0.95 * (len(yaw_sorted) - 1)))]
        coverage = max(0.0, float((max_progress[idx] - min_progress[idx]).item())) if active_count > 0 else 0.0
        backward = float(backward_progress[idx].item())
        forward = float(forward_progress[idx].item())
        backtracking_ratio = backward / max(forward + backward, 1e-6)
        action_delta_denom = active_action_delta_steps[idx].clamp_min(1.0)
        tip_vel_denom = brush_tip_vel_steps[idx].clamp_min(1.0)
        tip_accel_denom = brush_tip_accel_steps[idx].clamp_min(1.0)
        tip_jerk_denom = brush_tip_jerk_steps[idx].clamp_min(1.0)
        self_group_max = {
            name: float(values[idx].item())
            for name, values in self_collision_group_max.items()
            if torch.isfinite(values[idx])
        }
        self_group_margin = {
            name: float(values[idx].item())
            for name, values in self_collision_group_min_margin.items()
            if torch.isfinite(values[idx])
        }
        worst_group = max(self_group_max, key=self_group_max.get) if self_group_max else ""
        per_prior.append(
            {
                "env_id": idx,
                "prior_id": int(prior_ids[idx].item()),
                "active_steps": int(active_count),
                "eval_steps": int(eval_steps[idx].item()),
                "done_step": int(done_step[idx].item()),
                "survived": not bool(non_timeout_done[idx].item()),
                "contact_ratio": float((contact_hits[idx] / denom[idx]).item()),
                "row_ratio": float((row_hits[idx] / denom[idx]).item()),
                "target_row_ratio": float((target_row_hits[idx] / denom[idx]).item()),
                "combined_ratio": float((combined_hits[idx] / denom[idx]).item()),
                "target_combined_ratio": float((target_combined_hits[idx] / denom[idx]).item()),
                "ordered_anchor_hits": int(ordered_anchor_hits[idx].item()),
                "target_ordered_anchor_hits": int(target_ordered_anchor_hits[idx].item()),
                "n_successes": int(raw.n_successes[idx].item()) if hasattr(raw, "n_successes") else 0,
                "env_buffer_active_steps": int(env_buffer_active_steps[idx].item()),
                "env_buffer_contact_ratio": float(env_buffer_contact_ratio[idx].item()),
                "env_buffer_row_ratio": float(env_buffer_row_ratio[idx].item()),
                "env_buffer_combined_ratio": float(env_buffer_combined_ratio[idx].item()),
                "env_buffer_coverage": float(env_buffer_coverage[idx].item()),
                "env_buffer_next_anchor": int(env_buffer_next_anchor[idx].item()),
                "env_buffer_n_successes_peak": int(env_buffer_n_successes[idx].item()),
                "illegal_contact_ratio": float((illegal_hits[idx] / illegal_denom[idx]).item()),
                "torso_illegal_ratio": float((torso_illegal_hits[idx] / illegal_denom[idx]).item()),
                "right_hand_illegal_ratio": float((right_hand_illegal_hits[idx] / illegal_denom[idx]).item()),
                "left_arm_illegal_ratio": float((left_arm_illegal_hits[idx] / illegal_denom[idx]).item()),
                "wall_contact_termination_count": int(wall_contact_termination_hits[idx].item()),
                "illegal_contact_resets": int(wall_contact_termination_hits[idx].item()),
                "self_collision_termination_count": int(self_collision_termination_hits[idx].item()),
                "self_collision_resets": int(self_collision_termination_hits[idx].item()),
                "torso_below_termination_count": int(torso_below_termination_hits[idx].item()),
                "torso_angle_termination_count": int(torso_angle_termination_hits[idx].item()),
                "first_active_step": int(first_active_step[idx].item()),
                "min_root_z_m": float(min_root_z[idx].item()) if torch.isfinite(min_root_z[idx]) else 0.0,
                "min_root_cos_z": float(min_root_cos_z[idx].item()) if torch.isfinite(min_root_cos_z[idx]) else 0.0,
                "mean_wall_x_error_m": float((sum_wall_x_error[idx] / denom[idx]).item()),
                "mean_row_yz_error_m": float((sum_row_yz_error[idx] / denom[idx]).item()),
                "p95_row_yz_error_m": row_p95,
                "mean_reference_error_m": float((sum_reference_error[idx] / denom[idx]).item()),
                "mean_phase_error": float((sum_phase_error[idx] / denom[idx]).item()),
                "mean_torso_yaw_deg": float((sum_torso_yaw[idx] / denom[idx]).item()),
                "p95_torso_yaw_deg": yaw_p95,
                "mean_torso_upright_deg": float((sum_torso_upright[idx] / denom[idx]).item()),
                "mean_joint_prior_error_rad": float((sum_joint_prior_error[idx] / denom[idx]).item()),
                "mean_right_arm_joint_prior_error_rad": float((sum_right_arm_joint_prior_error[idx] / denom[idx]).item()),
                "mean_root_position_error_m": float((sum_root_position_error[idx] / denom[idx]).item()),
                "mean_root_orientation_error_deg": float((sum_root_orientation_error[idx] / denom[idx]).item()),
                "min_torso_clearance_m": float(min_torso_clearance[idx].item()) if active_count > 0 else 0.0,
                "min_nonbrush_clearance_m": float(min_nonbrush_clearance[idx].item()) if active_count > 0 else 0.0,
                "min_right_hand_clearance_m": float(min_right_hand_clearance[idx].item()) if active_count > 0 else 0.0,
                "min_left_arm_clearance_m": float(min_left_arm_clearance[idx].item()) if active_count > 0 else 0.0,
                "max_illegal_wall_contact_force_n": float(max_illegal_wall_contact_force[idx].item()),
                "max_self_collision_proxy_violation": float(max_self_collision_proxy_violation[idx].item()),
                "min_self_collision_margin_m": (
                    float(min_self_collision_margin[idx].item()) if eval_steps[idx].item() > 0 else 0.0
                ),
                "self_collision_worst_group": worst_group,
                "self_collision_group_max": self_group_max,
                "self_collision_group_min_margin_m": self_group_margin,
                "max_foot_slip_m": float(max_foot_slip[idx].item()),
                "mean_active_action_delta_l2": float((sum_active_action_delta_l2[idx] / action_delta_denom).item()),
                "max_active_action_delta_l2": float(max_active_action_delta_l2[idx].item()),
                "mean_active_right_arm_action_delta_l2": float(
                    (sum_active_right_arm_action_delta_l2[idx] / action_delta_denom).item()
                ),
                "max_active_right_arm_action_delta_l2": float(max_active_right_arm_action_delta_l2[idx].item()),
                "mean_brush_tip_speed_mps": float((sum_brush_tip_speed_mps[idx] / tip_vel_denom).item()),
                "max_brush_tip_speed_mps": float(max_brush_tip_speed_mps[idx].item()),
                "mean_brush_tip_accel_mps2": float((sum_brush_tip_accel_mps2[idx] / tip_accel_denom).item()),
                "max_brush_tip_accel_mps2": float(max_brush_tip_accel_mps2[idx].item()),
                "mean_brush_tip_jerk_mps3": float((sum_brush_tip_jerk_mps3[idx] / tip_jerk_denom).item()),
                "max_brush_tip_jerk_mps3": float(max_brush_tip_jerk_mps3[idx].item()),
                "max_progress": float(max_progress[idx].item()),
                "row_coverage": coverage,
                "backtracking_ratio": backtracking_ratio,
            }
        )

    valid = [row for row in per_prior if row["active_steps"] > 0]
    training_passes = [
        row["active_steps"] > 0
        and row["survived"]
        and row["n_successes"] > 0
        and row["illegal_contact_ratio"] == 0.0
        and row["self_collision_termination_count"] == 0
        and row["max_self_collision_proxy_violation"] <= 1e-6
        and row["min_torso_clearance_m"] >= args_cli.torso_wall_clearance
        and row["min_nonbrush_clearance_m"] >= args_cli.nonbrush_wall_clearance
        and row["mean_joint_prior_error_rad"] <= args_cli.joint_prior_error_threshold
        and row["mean_right_arm_joint_prior_error_rad"] <= args_cli.right_arm_prior_error_threshold
        and row["mean_root_orientation_error_deg"] <= args_cli.root_orientation_error_threshold_deg
        and row["mean_row_yz_error_m"] <= args_cli.row_yz_threshold
        and row["contact_ratio"] >= 0.60
        and row["row_ratio"] >= 0.60
        and row["combined_ratio"] >= 0.50
        and row["row_coverage"] >= args_cli.coverage_threshold
        and row["ordered_anchor_hits"] >= 3
        and row["max_foot_slip_m"] <= args_cli.foot_slip_threshold
        and row["p95_torso_yaw_deg"] <= args_cli.torso_yaw_threshold_deg
        and row["mean_torso_upright_deg"] <= args_cli.torso_upright_threshold_deg
        and row["backtracking_ratio"] <= args_cli.backtracking_threshold
        for row in per_prior
    ]
    target_passes = [
        row["active_steps"] > 0
        and row["survived"]
        and row["n_successes"] > 0
        and row["illegal_contact_ratio"] == 0.0
        and row["self_collision_termination_count"] == 0
        and row["max_self_collision_proxy_violation"] <= 1e-6
        and row["min_torso_clearance_m"] >= args_cli.target_torso_wall_clearance
        and row["min_nonbrush_clearance_m"] >= args_cli.target_nonbrush_wall_clearance
        and row["mean_joint_prior_error_rad"] <= args_cli.target_joint_prior_error_threshold
        and row["mean_right_arm_joint_prior_error_rad"] <= args_cli.target_right_arm_prior_error_threshold
        and row["mean_root_orientation_error_deg"] <= args_cli.target_root_orientation_error_threshold_deg
        and row["mean_row_yz_error_m"] <= args_cli.target_row_yz_threshold
        and row["p95_row_yz_error_m"] <= args_cli.target_row_yz_p95_threshold
        and row["target_combined_ratio"] >= 0.80
        and row["row_coverage"] >= args_cli.target_coverage_threshold
        and row["target_ordered_anchor_hits"] >= 3
        and row["max_foot_slip_m"] <= args_cli.target_foot_slip_threshold
        and row["mean_torso_yaw_deg"] <= args_cli.target_torso_yaw_threshold_deg
        and row["mean_torso_upright_deg"] <= args_cli.target_torso_upright_threshold_deg
        and row["backtracking_ratio"] <= args_cli.backtracking_threshold
        for row in per_prior
    ]
    for row, training_pass, target_pass in zip(per_prior, training_passes, target_passes):
        row["training_milestone_pass"] = bool(training_pass)
        row["acceptance_target_pass"] = bool(target_pass)

    prior_summary = summarize_27_prior_metrics(per_prior, training_passes, target_passes)
    suspicious_prior = select_suspicious_prior_for_visual_review(per_prior)
    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "action_mode": (
            "reference_actions" if args_cli.reference_actions else ("zero_residual" if args_cli.zero_actions else "checkpoint_policy")
        ),
        "action_smoothing_alpha": smoothing_alpha,
        "num_envs": num_envs,
        "num_steps": args_cli.num_steps,
        "contact_x_threshold_m": args_cli.contact_x_threshold,
        "row_yz_threshold_m": args_cli.row_yz_threshold,
        "target_row_yz_threshold_m": args_cli.target_row_yz_threshold,
        "clearance_thresholds_m": {
            "torso_training": args_cli.torso_wall_clearance,
            "torso_target": args_cli.target_torso_wall_clearance,
            "nonbrush_training": args_cli.nonbrush_wall_clearance,
            "nonbrush_target": args_cli.target_nonbrush_wall_clearance,
            "hand_training": args_cli.hand_wall_clearance,
            "hand_target": args_cli.target_hand_wall_clearance,
        },
        "wall_contact_force_threshold_n": args_cli.wall_contact_force_threshold,
        "wall_contact_near_margin_m": args_cli.wall_contact_near_margin,
        "torso_upright_threshold_deg": args_cli.torso_upright_threshold_deg,
        "target_torso_upright_threshold_deg": args_cli.target_torso_upright_threshold_deg,
        "pose_prior_thresholds": {
            "joint_training_rad": args_cli.joint_prior_error_threshold,
            "joint_target_rad": args_cli.target_joint_prior_error_threshold,
            "right_arm_training_rad": args_cli.right_arm_prior_error_threshold,
            "right_arm_target_rad": args_cli.target_right_arm_prior_error_threshold,
            "root_orientation_training_deg": args_cli.root_orientation_error_threshold_deg,
            "root_orientation_target_deg": args_cli.target_root_orientation_error_threshold_deg,
        },
        "upright_root_on_reset": _use_upright_root_on_reset(),
        "root_offset_z": float(args_cli.root_offset_z),
        "illegal_contact_definition": "x-axis normal contact force on near-wall torso/head/left-arm/legs/right-nonhand-arm links; right hand is legal",
        "self_collision_definition": "task-level proxy minimum distance across hand/forearm-torso, arm-arm, and hand-leg link groups; any negative margin is illegal",
        "survival_rate": _safe_mean([1.0 if row["survived"] else 0.0 for row in per_prior]),
        "expected_prior_count": prior_summary.expected_prior_count,
        "evaluated_prior_count": prior_summary.evaluated_prior_count,
        "dreamcontrol_style_success_count": prior_summary.dreamcontrol_style_success_count,
        "dreamcontrol_style_success_rate": prior_summary.dreamcontrol_style_success_rate,
        "training_milestone_pass_count": prior_summary.training_milestone_pass_count,
        "training_milestone_required_count": prior_summary.training_milestone_required_count,
        "training_milestone_pass_fixed": prior_summary.training_milestone_pass_fixed,
        "acceptance_target_pass_count": prior_summary.acceptance_target_pass_count,
        "acceptance_target_required_count": prior_summary.acceptance_target_required_count,
        "acceptance_target_pass_fixed": prior_summary.acceptance_target_pass_fixed,
        "suspicious_prior_id": suspicious_prior["prior_id"],
        "mean_done_step": _safe_mean([float(row["done_step"]) for row in per_prior]),
        "mean_contact_ratio": _safe_mean([row["contact_ratio"] for row in valid]),
        "mean_row_ratio": _safe_mean([row["row_ratio"] for row in valid]),
        "mean_target_row_ratio": _safe_mean([row["target_row_ratio"] for row in valid]),
        "mean_combined_ratio": _safe_mean([row["combined_ratio"] for row in valid]),
        "mean_target_combined_ratio": _safe_mean([row["target_combined_ratio"] for row in valid]),
        "mean_ordered_anchor_hits": _safe_mean([row["ordered_anchor_hits"] for row in valid]),
        "mean_target_ordered_anchor_hits": _safe_mean([row["target_ordered_anchor_hits"] for row in valid]),
        "mean_illegal_contact_ratio": _safe_mean([row["illegal_contact_ratio"] for row in valid]),
        "mean_torso_illegal_ratio": _safe_mean([row["torso_illegal_ratio"] for row in valid]),
        "mean_right_hand_illegal_ratio": _safe_mean([row["right_hand_illegal_ratio"] for row in valid]),
        "mean_left_arm_illegal_ratio": _safe_mean([row["left_arm_illegal_ratio"] for row in valid]),
        "total_wall_contact_terminations": sum(row["wall_contact_termination_count"] for row in per_prior),
        "total_self_collision_terminations": sum(row["self_collision_termination_count"] for row in per_prior),
        "total_torso_below_terminations": sum(row["torso_below_termination_count"] for row in per_prior),
        "total_torso_angle_terminations": sum(row["torso_angle_termination_count"] for row in per_prior),
        "entered_active_stroke_count": sum(1 for row in per_prior if row["first_active_step"] >= 0),
        "min_root_z_m": min([row["min_root_z_m"] for row in per_prior], default=0.0),
        "min_root_cos_z": min([row["min_root_cos_z"] for row in per_prior], default=0.0),
        "mean_wall_x_error_m": _safe_mean([row["mean_wall_x_error_m"] for row in valid]),
        "mean_row_yz_error_m": _safe_mean([row["mean_row_yz_error_m"] for row in valid]),
        "max_p95_row_yz_error_m": max([row["p95_row_yz_error_m"] for row in valid], default=0.0),
        "mean_reference_error_m": _safe_mean([row["mean_reference_error_m"] for row in valid]),
        "mean_phase_error": _safe_mean([row["mean_phase_error"] for row in valid]),
        "mean_max_progress": _safe_mean([row["max_progress"] for row in valid]),
        "mean_row_coverage": _safe_mean([row["row_coverage"] for row in valid]),
        "mean_backtracking_ratio": _safe_mean([row["backtracking_ratio"] for row in valid]),
        "max_backtracking_ratio": max([row["backtracking_ratio"] for row in valid], default=0.0),
        "min_torso_clearance_m": min([row["min_torso_clearance_m"] for row in valid], default=0.0),
        "min_nonbrush_clearance_m": min([row["min_nonbrush_clearance_m"] for row in valid], default=0.0),
        "min_right_hand_clearance_m": min([row["min_right_hand_clearance_m"] for row in valid], default=0.0),
        "min_left_arm_clearance_m": min([row["min_left_arm_clearance_m"] for row in valid], default=0.0),
        "max_illegal_wall_contact_force_n": max([row["max_illegal_wall_contact_force_n"] for row in per_prior], default=0.0),
        "max_foot_slip_m": max([row["max_foot_slip_m"] for row in valid], default=0.0),
        "mean_active_action_delta_l2": _safe_mean([row["mean_active_action_delta_l2"] for row in valid]),
        "max_active_action_delta_l2": max([row["max_active_action_delta_l2"] for row in valid], default=0.0),
        "mean_active_right_arm_action_delta_l2": _safe_mean(
            [row["mean_active_right_arm_action_delta_l2"] for row in valid]
        ),
        "max_active_right_arm_action_delta_l2": max(
            [row["max_active_right_arm_action_delta_l2"] for row in valid], default=0.0
        ),
        "mean_brush_tip_speed_mps": _safe_mean([row["mean_brush_tip_speed_mps"] for row in valid]),
        "max_brush_tip_speed_mps": max([row["max_brush_tip_speed_mps"] for row in valid], default=0.0),
        "mean_brush_tip_accel_mps2": _safe_mean([row["mean_brush_tip_accel_mps2"] for row in valid]),
        "max_brush_tip_accel_mps2": max([row["max_brush_tip_accel_mps2"] for row in valid], default=0.0),
        "mean_brush_tip_jerk_mps3": _safe_mean([row["mean_brush_tip_jerk_mps3"] for row in valid]),
        "max_brush_tip_jerk_mps3": max([row["max_brush_tip_jerk_mps3"] for row in valid], default=0.0),
        "mean_torso_yaw_deg": _safe_mean([row["mean_torso_yaw_deg"] for row in valid]),
        "max_p95_torso_yaw_deg": max([row["p95_torso_yaw_deg"] for row in valid], default=0.0),
        "mean_torso_upright_deg": _safe_mean([row["mean_torso_upright_deg"] for row in valid]),
        "mean_joint_prior_error_rad": _safe_mean([row["mean_joint_prior_error_rad"] for row in valid]),
        "mean_right_arm_joint_prior_error_rad": _safe_mean(
            [row["mean_right_arm_joint_prior_error_rad"] for row in valid]
        ),
        "mean_root_position_error_m": _safe_mean([row["mean_root_position_error_m"] for row in valid]),
        "mean_root_orientation_error_deg": _safe_mean([row["mean_root_orientation_error_deg"] for row in valid]),
        "max_self_collision_proxy_violation": max(
            [row["max_self_collision_proxy_violation"] for row in valid], default=0.0
        ),
        "min_self_collision_margin_m": min([row["min_self_collision_margin_m"] for row in valid], default=0.0),
        "training_milestone_pass_rate": prior_summary.training_milestone_pass_rate,
        "acceptance_target_pass_rate": prior_summary.acceptance_target_pass_rate,
    }

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "per_prior": per_prior}
    if trace_prior_ids:
        payload["trace"] = trace_rows
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_prior[0].keys()))
        writer.writeheader()
        writer.writerows(per_prior)

    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote {output}")
    print(f"[INFO] Wrote {csv_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
