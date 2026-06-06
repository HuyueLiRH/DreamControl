#!/usr/bin/env python3
"""Play an RSL-RL checkpoint with a fixed wall-brush review camera."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


def _triple(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats")
    return tuple(parts)


def _resolution(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers")
    return tuple(parts)


parser = argparse.ArgumentParser(description="Play a wall-brush RSL-RL checkpoint with a fixed camera.")
parser.add_argument("--video", action="store_true", default=True, help="Record a video.")
parser.add_argument("--video_length", type=int, default=220, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Wall-Brush-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use a published pretrained checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument("--camera_eye", type=_triple, default=(-0.85, -1.65, 1.22), help="Camera eye as x,y,z.")
parser.add_argument("--camera_lookat", type=_triple, default=(0.36, 0.00, 0.92), help="Camera target as x,y,z.")
parser.add_argument("--camera_resolution", type=_resolution, default=(1600, 1000), help="Camera resolution as w,h.")
parser.add_argument("--view_name", type=str, default="wall_brush_fixed_view", help="Output video subfolder name.")
parser.add_argument("--show_wall_brush_markers", action="store_true", default=True, help="Show brush-tip, row, trail, and illegal-contact markers.")
parser.add_argument("--hide_wall_brush_markers", action="store_true", default=False, help="Disable review markers for Newton/renderer compatibility.")
parser.add_argument("--disable_review_wall_material", action="store_true", default=False, help="Do not override the wall material.")
parser.add_argument("--prior_id", type=int, default=0, help="Deterministic prior ID for qualitative review. Use -1 for random reset.")
parser.add_argument("--trail_points", type=int, default=120, help="Maximum brush-tip trail points to display.")
parser.add_argument("--illegal_clearance", type=float, default=0.05, help="Red-marker clearance threshold for non-brush body links.")
parser.add_argument("--torso_clearance", type=float, default=0.10, help="Red-marker clearance threshold for torso/pelvis/head links.")
parser.add_argument("--wall_contact_force_threshold", type=float, default=1.0, help="Red-marker threshold for illegal non-brush wall contact in Newtons.")
parser.add_argument("--wall_contact_near_margin", type=float, default=0.22, help="Only show contact-force red markers for bodies this close to the wall plane.")
parser.add_argument("--review_wall_color", type=_triple, default=(0.10, 0.50, 1.00), help="Visible review-wall RGB color.")
parser.add_argument("--review_wall_opacity", type=float, default=0.85, help="Visible review-wall opacity.")
parser.add_argument(
    "--upright_root_on_reset",
    choices=("auto", "always", "never"),
    default="auto",
    help="Use an upright root orientation when locking deterministic priors. Auto enables this for stance tasks.",
)
parser.add_argument("--root_offset_z", type=float, default=0.0, help="Additional z offset for deterministic prior reset.")
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

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder


installed_version = metadata.version("rsl-rl-lib")


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


BRUSH_LINK = "right_hand_index_1_link"
RIGHT_HAND_NAMES = {
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
}
TORSO_HEAD_NAMES = [
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
]
WALL_HALF_Y = 0.425
WALL_HALF_Z = 0.375


def _use_upright_root_on_reset() -> bool:
    if args_cli.upright_root_on_reset == "always":
        return True
    if args_cli.upright_root_on_reset == "never":
        return False
    return "Stance" in args_cli.task


def _policy_obs(env):
    obs_result = env.get_observations()
    return obs_result[0] if isinstance(obs_result, tuple) else obs_result


def _body_ids(asset: Articulation, names: list[str], device) -> torch.Tensor:
    ids = [asset.body_names.index(name) for name in names if name in asset.body_names]
    if not ids:
        raise ValueError(f"None of the requested body names exist: {names}")
    return torch.tensor(ids, device=device, dtype=torch.long)


def _reset_to_prior_id(raw_env, prior_id: int):
    if prior_id < 0 or not hasattr(raw_env, "motion_lib"):
        return
    device = raw_env.device
    env_ids = torch.arange(raw_env.scene.num_envs, device=device)
    motion_ids = torch.full((raw_env.scene.num_envs,), int(prior_id) % int(raw_env.total_motions), device=device, dtype=torch.long)
    raw_env.motion_ids[env_ids] = motion_ids
    raw_env.start_motion_times[env_ids] = 0.0
    raw_env.episode_length_buf[env_ids] = 0

    motion_res = raw_env.motion_lib.get_motion_state(motion_ids, torch.zeros(raw_env.scene.num_envs, device=device))
    asset: Articulation = raw_env.scene["robot"]
    joint_ids = torch.tensor([asset.joint_names.index(name) for name in JointNamesOrder], device=device)
    joint_pos = _as_torch(asset.data.default_joint_pos)[env_ids].clone()
    joint_vel = _as_torch(asset.data.default_joint_vel)[env_ids].clone()
    joint_pos[:, joint_ids] = motion_res["dof_pos"]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    positions = motion_res["root_pos"] + raw_env.scene.env_origins[env_ids]
    positions[:, 2] += float(args_cli.root_offset_z)
    if _use_upright_root_on_reset():
        orientations = torch.zeros((raw_env.scene.num_envs, 4), device=device)
        orientations[:, 0] = 1.0
    else:
        orientations = motion_res["root_rot"]
    velocities = torch.zeros((raw_env.scene.num_envs, 6), device=device)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    raw_env.scene.write_data_to_sim()
    raw_env.sim.forward()
    raw_env.scene.update(raw_env.step_dt)


def _create_markers():
    wall_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/WallBrush/review_wall",
            markers={
                "wall": sim_utils.SphereCfg(
                    radius=0.011,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.55, 1.0)),
                )
            },
        )
    )
    tip_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/WallBrush/brush_tip",
            markers={
                "tip": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            },
        )
    )
    row_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/WallBrush/target_row",
            markers={
                "row": sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
                ),
                "ref": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.0)),
                ),
                "anchor": sim_utils.SphereCfg(
                    radius=0.032,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
                ),
            },
        )
    )
    trail_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/WallBrush/trail",
            markers={
                "trail": sim_utils.SphereCfg(
                    radius=0.011,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 1.0)),
                )
            },
        )
    )
    illegal_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/WallBrush/illegal_contact",
            markers={
                "illegal": sim_utils.SphereCfg(
                    radius=0.035,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            },
        )
    )
    return wall_marker, tip_marker, row_marker, trail_marker, illegal_marker


def _motion_state(raw_env):
    motion_times = raw_env.episode_length_buf * raw_env.step_dt + raw_env.start_motion_times.to(
        device=raw_env.device, dtype=torch.float32
    )
    return raw_env.motion_lib.get_motion_state(raw_env.motion_ids, motion_times)


def action_mode_reference(raw_env) -> torch.Tensor:
    action_term = raw_env.action_manager.get_term("joint_pos")
    motion_res = _motion_state(raw_env)
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


def _wall_contact_sensor(raw_env):
    try:
        return raw_env.scene.sensors["contact_forces"]
    except KeyError:
        return None


def _illegal_wall_contact_positions(raw_env, asset: Articulation, motion_res) -> torch.Tensor:
    sensor = _wall_contact_sensor(raw_env)
    if sensor is None or sensor.data.net_forces_w is None:
        return torch.empty((0, 3), device=raw_env.device)

    sensor_names = sensor.body_names
    sensor_ids = [idx for idx, name in enumerate(sensor_names) if name not in RIGHT_HAND_NAMES and name in asset.body_names]
    if not sensor_ids:
        return torch.empty((0, 3), device=raw_env.device)

    asset_ids = torch.tensor([asset.body_names.index(sensor_names[idx]) for idx in sensor_ids], device=raw_env.device)
    body_state_w = _as_torch(asset.data.body_state_w)
    body_pos = body_state_w[0, asset_ids, :3]
    body_pos_env = body_pos - raw_env.scene.env_origins[0].unsqueeze(0)
    clearance = motion_res["wall_mid"][0, 0] - body_pos_env[:, 0]
    near_wall = clearance < args_cli.wall_contact_near_margin
    forces = torch.abs(_as_torch(sensor.data.net_forces_w)[0, sensor_ids, 0])
    illegal = (forces > args_cli.wall_contact_force_threshold) & near_wall
    if not torch.any(illegal):
        return torch.empty((0, 3), device=raw_env.device)

    return body_pos[illegal]


def _update_markers(raw_env, markers, trail: list[torch.Tensor]):
    wall_marker, tip_marker, row_marker, trail_marker, illegal_marker = markers
    asset: Articulation = raw_env.scene["robot"]
    motion_res = _motion_state(raw_env)
    body_id = asset.body_names.index(BRUSH_LINK)
    env_origin = raw_env.scene.env_origins[0]
    wall_mid = motion_res["wall_mid"][0] + env_origin
    wall_y = torch.linspace(wall_mid[1] - WALL_HALF_Y, wall_mid[1] + WALL_HALF_Y, 13, device=raw_env.device)
    wall_z = torch.linspace(wall_mid[2] - WALL_HALF_Z, wall_mid[2] + WALL_HALF_Z, 11, device=raw_env.device)
    wall_grid_y, wall_grid_z = torch.meshgrid(wall_y, wall_z, indexing="ij")
    wall_grid = torch.stack(
        [
            torch.full_like(wall_grid_y, wall_mid[0] - 0.018),
            wall_grid_y,
            wall_grid_z,
        ],
        dim=-1,
    ).reshape(-1, 3)
    wall_marker.visualize(translations=wall_grid)

    body_state_w = _as_torch(asset.data.body_state_w)
    brush_tip_world = body_state_w[0, body_id, :3]
    tip_marker.visualize(translations=brush_tip_world.unsqueeze(0))

    start = motion_res["wall_start"][0] + env_origin
    mid = motion_res["wall_mid"][0] + env_origin
    end = motion_res["wall_end"][0] + env_origin
    ref = motion_res["brush_tip_pos"][0] + env_origin
    row_points = torch.linspace(0.0, 1.0, 31, device=raw_env.device).unsqueeze(1) * (end - start).unsqueeze(0) + start
    row_points[:, 0] -= 0.01
    anchors = torch.stack([start, mid, end], dim=0)
    anchors[:, 0] -= 0.02
    row_translations = torch.cat([row_points, ref.unsqueeze(0), anchors], dim=0)
    row_indices = torch.cat(
        [
            torch.zeros(row_points.shape[0], dtype=torch.long, device=raw_env.device),
            torch.ones(1, dtype=torch.long, device=raw_env.device),
            torch.full((3,), 2, dtype=torch.long, device=raw_env.device),
        ]
    )
    row_marker.visualize(translations=row_translations, marker_indices=row_indices)

    trail.append(brush_tip_world.detach().clone())
    if len(trail) > args_cli.trail_points:
        del trail[: len(trail) - args_cli.trail_points]
    trail_marker.visualize(translations=torch.stack(trail, dim=0))

    illegal_positions = _illegal_wall_contact_positions(raw_env, asset, motion_res)
    if illegal_positions.shape[0] == 0 and _wall_contact_sensor(raw_env) is None:
        body_pos = body_state_w[0, :, :3]
        body_pos_env = body_pos - env_origin.unsqueeze(0)
        wall_x = motion_res["wall_mid"][0, 0]
        clearance = wall_x - body_pos_env[:, 0]
        torso_ids = _body_ids(asset, TORSO_HEAD_NAMES, raw_env.device)
        torso_mask = torch.zeros_like(clearance, dtype=torch.bool)
        torso_mask[torso_ids] = True
        illegal_mask = (clearance < args_cli.illegal_clearance) | (torso_mask & (clearance < args_cli.torso_clearance))
        illegal_positions = body_pos[illegal_mask]
    if illegal_positions.shape[0] == 0:
        illegal_positions = torch.tensor([[0.0, 0.0, -100.0]], device=raw_env.device)
    illegal_marker.visualize(translations=illegal_positions)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.viewer.eye = args_cli.camera_eye
    env_cfg.viewer.lookat = args_cli.camera_lookat
    env_cfg.viewer.resolution = args_cli.camera_resolution
    env_cfg.viewer.origin_type = "world"
    if hasattr(env_cfg.scene, "wall") and not args_cli.disable_review_wall_material:
        env_cfg.scene.wall.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=args_cli.review_wall_color,
            metallic=0.0,
            opacity=args_cli.review_wall_opacity,
        )

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Fixed review camera eye={env_cfg.viewer.eye}, lookat={env_cfg.viewer.lookat}, resolution={env_cfg.viewer.resolution}")
    print(f"[INFO] Review wall color={args_cli.review_wall_color}, opacity={args_cli.review_wall_opacity}")
    print(f"[INFO] Upright root on deterministic reset={_use_upright_root_on_reset()}")
    print(f"[INFO] Root z offset on deterministic reset={float(args_cli.root_offset_z):.3f}")
    smoothing_alpha = max(0.0, min(1.0, float(args_cli.action_smoothing_alpha)))
    print(f"[INFO] Action smoothing alpha={smoothing_alpha:.3f}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", args_cli.view_name),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording fixed-view wall-brush video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    _reset_to_prior_id(env.unwrapped, args_cli.prior_id)

    if args_cli.reference_actions:
        print("[INFO]: Using action_mode_reference instead of a checkpoint policy.")
        runner = None
        policy = None
    else:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = _policy_obs(env)
    dt = env.unwrapped.step_dt
    timestep = 0
    markers = _create_markers() if args_cli.show_wall_brush_markers and not args_cli.hide_wall_brush_markers else None
    trail: list[torch.Tensor] = []
    smoothed_actions = None
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if markers is not None:
                _update_markers(env.unwrapped, markers, trail)
            if args_cli.reference_actions:
                actions = action_mode_reference(env.unwrapped)
            else:
                actions = policy(obs)
            actions = smooth_actions(actions, smoothed_actions, smoothing_alpha)
            smoothed_actions = actions.clone()
            obs, _, dones, _ = env.step(actions)
            if smoothed_actions is not None:
                smoothed_actions[dones.bool()] = 0.0
            if markers is not None:
                _update_markers(env.unwrapped, markers, trail)
            if policy is not None and version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
