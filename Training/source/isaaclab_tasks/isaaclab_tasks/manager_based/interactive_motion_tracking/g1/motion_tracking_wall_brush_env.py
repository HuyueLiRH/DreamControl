# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch

from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab_assets import G1_MINIMAL_CFG, G1_MINIMAL_CFG_FIXED_BASE  # isort: skip
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_interactive_base import (
    ActionsCfg as ActionsCfgBase,
    EventCfg as EventCfgBase,
    G1InteractiveBaseEnvCfg,
    G1Rewards as G1RewardsBase,
    MySceneCfg as MySceneCfgBase,
    TerminationsCfg as TerminationsCfgBase,
)
from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import (
    KEYPTS_MASK,
    JOINTS_MASK,
    current_time_enc,
    joint_deviation_ref_l1,
    keypts_deviation_ref_l2,
    orientation_tracking_error,
    position_tracking_error,
    reset_root_state_for_motion,
    root_angle_below_threshold,
    root_below_threshold,
    target_ref,
)
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder, JointNamesOrder_UB


VISUALIZE_MARKERS = False
BRUSH_LINK = "right_hand_index_1_link"
TORSO_HEAD_LINKS = [
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
]
# The G1 head/logo meshes may be fixed under the torso in PhysX. Use
# torso-frame proxy points so head/chest wall support resets even when contact
# forces are silent or fixed child links are not exposed as articulation bodies.
TORSO_HEAD_WALL_PROXY_OFFSETS = [
    [0.12, 0.00, -0.08],
    [0.14, 0.00, 0.08],
    [0.14, 0.08, 0.08],
    [0.14, -0.08, 0.08],
    [0.16, 0.00, 0.25],
    [0.18, 0.00, 0.38],
]
LEFT_HAND_LINKS = [
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
LEFT_ARM_LINKS = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    *LEFT_HAND_LINKS,
]
RIGHT_HAND_LINKS = [
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
RIGHT_HAND_NONBRUSH_LINKS = [
    link for link in RIGHT_HAND_LINKS if link != BRUSH_LINK
]
RIGHT_ARM_LINKS = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    *RIGHT_HAND_LINKS,
]
RIGHT_NONHAND_ARM_LINKS = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
]
LEG_LINKS = [
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
NONBRUSH_LINKS = TORSO_HEAD_LINKS + LEFT_ARM_LINKS + RIGHT_NONHAND_ARM_LINKS + LEG_LINKS
ILLEGAL_WALL_CONTACT_LINKS = NONBRUSH_LINKS
SELF_COLLISION_PAIR_GROUPS = [
    ("right_hand_torso", RIGHT_HAND_LINKS, TORSO_HEAD_LINKS, 0.075),
    ("left_hand_torso", LEFT_HAND_LINKS, TORSO_HEAD_LINKS, 0.075),
    ("right_forearm_torso", RIGHT_NONHAND_ARM_LINKS, TORSO_HEAD_LINKS, 0.065),
    ("left_forearm_torso", ["left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link"], TORSO_HEAD_LINKS, 0.065),
    ("hands_cross", RIGHT_HAND_LINKS, LEFT_HAND_LINKS, 0.050),
    ("right_arm_left_arm", RIGHT_NONHAND_ARM_LINKS, LEFT_ARM_LINKS, 0.055),
    ("hands_legs", RIGHT_HAND_LINKS + LEFT_HAND_LINKS, LEG_LINKS, 0.045),
]
RIGHT_HAND_LEG_COLLISION_PAIR_GROUPS = [
    ("right_hand_legs", RIGHT_HAND_LINKS, LEG_LINKS, 0.045),
]
SELF_COLLISION_HARD_RESET_PAIR_GROUPS = [
    group for group in SELF_COLLISION_PAIR_GROUPS if group[0] != "hands_legs"
]
FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]
FOOT_KEYPOINT_IDS = [7, 13]
LOWER_BODY_JOINTS_MASK = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
UPPER_BODY_JOINTS_MASK = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
RIGHT_ARM_JOINTS_MASK = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
AGILE_LOWER_BODY_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
AGILE_TEACHER_JOINT_OBS_ORDER = [
    *AGILE_LOWER_BODY_JOINTS,
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
AGILE_TEACHER_POLICY_PATH = (
    "/root/autodl-tmp/WBC-AGILE/agile/data/policy/velocity_height_g1/"
    "unitree_g1_velocity_height_teacher.pt"
)
AGILE_LOWER_BODY_POLICY_OUTPUT_SCALE = {
    ".*_hip_yaw_joint": 0.22,
    ".*_hip_roll_joint": 0.22,
    ".*_hip_pitch_joint": 0.22,
    ".*_knee_joint": 0.17375,
    ".*_ankle_pitch_joint": 0.625,
    ".*_ankle_roll_joint": 0.625,
}
RIGHT_ARM_WITH_WAIST_JOINTS_MASK = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
RIGHT_ARM_WITH_TORSO_KEYPTS_MASK = [
    0.0,  # pelvis
    0.0,  # pelvis_contour_link
    0.0,  # left_hip_pitch_link
    0.0,  # left_hip_roll_link
    0.0,  # left_hip_yaw_link
    0.0,  # left_knee_link
    0.0,  # left_ankle_pitch_link
    0.0,  # left_ankle_roll_link
    0.0,  # right_hip_pitch_link
    0.0,  # right_hip_roll_link
    0.0,  # right_hip_yaw_link
    0.0,  # right_knee_link
    0.0,  # right_ankle_pitch_link
    0.0,  # right_ankle_roll_link
    1.0,  # waist_yaw_link
    1.0,  # waist_roll_link
    1.0,  # torso_link
    1.0,  # logo_link
    1.0,  # head_link
    1.0,  # waist_support_link
    1.0,  # imu_link
    1.0,  # d435_link
    1.0,  # mid360_link
    0.0,  # left_shoulder_pitch_link
    0.0,  # left_shoulder_roll_link
    0.0,  # left_shoulder_yaw_link
    0.0,  # left_elbow_link
    0.0,  # left_wrist_roll_link
    0.0,  # left_wrist_pitch_link
    0.0,  # left_wrist_yaw_link
    0.0,  # left_rubber_hand
    1.0,  # right_shoulder_pitch_link
    1.0,  # right_shoulder_roll_link
    1.0,  # right_shoulder_yaw_link
    1.0,  # right_elbow_link
    1.0,  # right_wrist_roll_link
    1.0,  # right_wrist_pitch_link
    1.0,  # right_wrist_yaw_link
    1.0,  # right_rubber_hand
]
BUTTONPRESS_TRACKING_JOINTS_MASK = [
    1.0,  # left_hip_pitch_joint
    1.0,  # left_hip_roll_joint
    1.0,  # left_hip_yaw_joint
    1.0,  # left_knee_joint
    0.0,  # left_ankle_pitch_joint
    0.0,  # left_ankle_roll_joint
    1.0,  # right_hip_pitch_joint
    1.0,  # right_hip_roll_joint
    1.0,  # right_hip_yaw_joint
    1.0,  # right_knee_joint
    0.0,  # right_ankle_pitch_joint
    0.0,  # right_ankle_roll_joint
    1.0,  # waist_yaw_joint
    1.0,  # left_shoulder_pitch_joint
    1.0,  # left_shoulder_roll_joint
    1.0,  # left_shoulder_yaw_joint
    1.0,  # left_elbow_joint
    1.0,  # left_wrist_roll_joint
    1.0,  # left_wrist_pitch_joint
    1.0,  # left_wrist_yaw_joint
    1.0,  # right_shoulder_pitch_joint
    1.0,  # right_shoulder_roll_joint
    1.0,  # right_shoulder_yaw_joint
    1.0,  # right_elbow_joint
    1.0,  # right_wrist_roll_joint
    1.0,  # right_wrist_pitch_joint
    1.0,  # right_wrist_yaw_joint
]
BUTTONPRESS_TRACKING_KEYPTS_MASK = [
    1.0,  # pelvis
    1.0,  # pelvis_contour_link
    1.0,  # left_hip_pitch_link
    1.0,  # left_hip_roll_link
    1.0,  # left_hip_yaw_link
    2.0,  # left_knee_link
    2.0,  # left_ankle_pitch_link
    2.0,  # left_ankle_roll_link
    1.0,  # right_hip_pitch_link
    1.0,  # right_hip_roll_link
    1.0,  # right_hip_yaw_link
    2.0,  # right_knee_link
    2.0,  # right_ankle_pitch_link
    2.0,  # right_ankle_roll_link
    1.0,  # waist_yaw_link
    1.0,  # waist_roll_link
    1.0,  # torso_link
    1.0,  # logo_link
    1.0,  # head_link
    1.0,  # waist_support_link
    1.0,  # imu_link
    1.0,  # d435_link
    1.0,  # mid360_link
    1.0,  # left_shoulder_pitch_link
    1.0,  # left_shoulder_roll_link
    1.0,  # left_shoulder_yaw_link
    1.0,  # left_elbow_link
    1.0,  # left_wrist_roll_link
    1.0,  # left_wrist_pitch_link
    1.0,  # left_wrist_yaw_link
    1.0,  # left_rubber_hand
    1.0,  # right_shoulder_pitch_link
    1.0,  # right_shoulder_roll_link
    1.0,  # right_shoulder_yaw_link
    1.0,  # right_elbow_link
    1.0,  # right_wrist_roll_link
    1.0,  # right_wrist_pitch_link
    1.0,  # right_wrist_yaw_link
    0.0,  # right_rubber_hand
]
UPPER_BODY_WITH_WAIST_JOINTS_MASK = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
WALL_BRUSH_CURRICULUM_PRIOR_WEIGHTS = [
    2.0,
    2.0,
    2.0,
    4.0,
    4.0,
    4.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    4.0,
    4.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    4.0,
    4.0,
    2.0,
    2.0,
    2.0,
]


def _has_motion_lib(env) -> bool:
    return hasattr(env, "motion_lib") and hasattr(env, "motion_ids") and hasattr(env, "start_motion_times")


def _motion_state(env, time_offset: float = 0.0):
    if not _has_motion_lib(env):
        return None
    motion_times = (
        env.episode_length_buf * env.step_dt
        + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
        + time_offset
    )
    return env.motion_lib.get_motion_state(env.motion_ids, motion_times)


def _stance_motion_state(env):
    if not _has_motion_lib(env):
        return None
    motion_times = env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    return env.motion_lib.get_motion_state(env.motion_ids, motion_times)


class WallBrushMotionResidualJointPositionAction(JointPositionAction):
    """Joint-position residual action centered on the current wall-brush reference."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        missing = [name for name in self._joint_names if name not in JointNamesOrder]
        if missing:
            raise ValueError(f"Wall-brush residual action joints must be in JointNamesOrder. Missing: {missing}")
        self._motion_joint_order_ids = torch.tensor(
            [JointNamesOrder.index(name) for name in self._joint_names], device=self.device, dtype=torch.long
        )

    def _reference_joint_pos(self) -> torch.Tensor:
        reference_mode = getattr(self.cfg, "reference_mode", "current")
        if reference_mode == "stance":
            motion_res = _stance_motion_state(self._env)
        else:
            motion_res = _motion_state(self._env, getattr(self.cfg, "reference_time_offset", 0.0))
        if motion_res is None:
            return self._asset.data.default_joint_pos[:, self._joint_ids]
        return motion_res["dof_pos"][:, self._motion_joint_order_ids]

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._reference_joint_pos()
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


class WallBrushAgileLowerBodyAction(ActionTerm):
    """Frozen AGILE lower-body policy action driven by a fixed velocity-height command."""

    _AGILE_TEACHER_INPUT_DIM = 83
    _AGILE_TEACHER_OUTPUT_DIM = 12

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=True)
        if len(self._joint_names) != self._AGILE_TEACHER_OUTPUT_DIM:
            raise ValueError(
                f"AGILE lower body expects {self._AGILE_TEACHER_OUTPUT_DIM} joints, got {len(self._joint_names)}"
            )

        self._policy = torch.jit.load(self.cfg.policy_path, map_location=self.device)
        self._policy.eval()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

        self._fixed_command = torch.tensor(self.cfg.fixed_command, device=self.device, dtype=torch.float32).view(1, 4)
        self._fixed_command = self._fixed_command.repeat(self.num_envs, 1)
        self._raw_actions = torch.zeros(self.num_envs, self._AGILE_TEACHER_OUTPUT_DIM, device=self.device)
        self._processed_actions = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._teacher_obs_joint_ids = []
        self._teacher_obs_valid_ids = []
        joint_name_to_id = {name: idx for idx, name in enumerate(self._asset.joint_names)}
        for obs_id, joint_name in enumerate(AGILE_TEACHER_JOINT_OBS_ORDER):
            if joint_name in joint_name_to_id:
                self._teacher_obs_joint_ids.append(joint_name_to_id[joint_name])
                self._teacher_obs_valid_ids.append(obs_id)
        self._teacher_obs_joint_ids = torch.tensor(self._teacher_obs_joint_ids, device=self.device, dtype=torch.long)
        self._teacher_obs_valid_ids = torch.tensor(self._teacher_obs_valid_ids, device=self.device, dtype=torch.long)

        self._policy_output_scale = torch.ones(self.num_envs, self._AGILE_TEACHER_OUTPUT_DIM, device=self.device)
        for idx, joint_name in enumerate(self._joint_names):
            if "hip" in joint_name:
                self._policy_output_scale[:, idx] = 0.22
            elif "knee" in joint_name:
                self._policy_output_scale[:, idx] = 0.17375
            elif "ankle" in joint_name:
                self._policy_output_scale[:, idx] = 0.625
        self._policy_output_offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _teacher_joint_observation(self, velocity: bool = False) -> torch.Tensor:
        values = torch.zeros(self.num_envs, len(AGILE_TEACHER_JOINT_OBS_ORDER), device=self.device)
        if self._teacher_obs_joint_ids.numel() == 0:
            return values
        if velocity:
            values[:, self._teacher_obs_valid_ids] = self._asset.data.joint_vel[:, self._teacher_obs_joint_ids] * 0.1
        else:
            values[:, self._teacher_obs_valid_ids] = (
                self._asset.data.joint_pos[:, self._teacher_obs_joint_ids]
                - self._asset.data.default_joint_pos[:, self._teacher_obs_joint_ids]
            )
        return values

    def _teacher_input(self) -> torch.Tensor:
        obs = torch.cat(
            [
                self._fixed_command,
                self._asset.data.root_lin_vel_b,
                self._asset.data.root_ang_vel_b,
                self._asset.data.projected_gravity_b,
                self._teacher_joint_observation(velocity=False),
                self._teacher_joint_observation(velocity=True),
                self._raw_actions,
            ],
            dim=-1,
        )
        if obs.shape[-1] != self._AGILE_TEACHER_INPUT_DIM:
            raise RuntimeError(f"AGILE teacher input must be 83D, got {obs.shape[-1]}")
        return obs

    def process_actions(self, actions: torch.Tensor):
        del actions
        with torch.inference_mode():
            joint_actions = self._policy(self._teacher_input())
        if joint_actions.shape[-1] != self._AGILE_TEACHER_OUTPUT_DIM:
            raise RuntimeError(f"AGILE teacher output must be 12D, got {joint_actions.shape[-1]}")
        self._raw_actions[:] = joint_actions
        self._processed_actions = joint_actions * self._policy_output_scale + self._policy_output_offset

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._asset.data.default_joint_pos[env_ids, :][:, self._joint_ids]
        if hasattr(self._policy, "reset"):
            self._policy.reset()


def _zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.scene.num_envs, device=env.device)


def _ensure_wall_brush_success_buffers(env):
    if hasattr(env, "wall_brush_active_steps"):
        return
    count_shape = (env.scene.num_envs,)
    env.wall_brush_active_steps = torch.zeros(count_shape, device=env.device, dtype=torch.float32)
    env.wall_brush_contact_steps = torch.zeros(count_shape, device=env.device, dtype=torch.float32)
    env.wall_brush_row_steps = torch.zeros(count_shape, device=env.device, dtype=torch.float32)
    env.wall_brush_combined_steps = torch.zeros(count_shape, device=env.device, dtype=torch.float32)
    env.wall_brush_min_phase = torch.full(count_shape, float("inf"), device=env.device)
    env.wall_brush_max_phase = torch.full(count_shape, float("-inf"), device=env.device)
    env.wall_brush_next_anchor = torch.zeros(count_shape, device=env.device, dtype=torch.long)
    env.wall_brush_prev_anchor_dist = torch.full(count_shape, float("inf"), device=env.device)
    env.wall_brush_pending_success = torch.zeros(count_shape, device=env.device, dtype=torch.bool)
    env.wall_brush_countable_success = torch.zeros(count_shape, device=env.device, dtype=torch.bool)
    env.wall_brush_invalidated_success = torch.zeros(count_shape, device=env.device, dtype=torch.bool)


def reset_wall_brush_success_buffers(env, env_ids: torch.Tensor):
    _ensure_wall_brush_success_buffers(env)
    env.wall_brush_active_steps[env_ids] = 0.0
    env.wall_brush_contact_steps[env_ids] = 0.0
    env.wall_brush_row_steps[env_ids] = 0.0
    env.wall_brush_combined_steps[env_ids] = 0.0
    env.wall_brush_min_phase[env_ids] = float("inf")
    env.wall_brush_max_phase[env_ids] = float("-inf")
    env.wall_brush_next_anchor[env_ids] = 0
    env.wall_brush_prev_anchor_dist[env_ids] = float("inf")
    env.wall_brush_pending_success[env_ids] = False
    env.wall_brush_countable_success[env_ids] = False
    env.wall_brush_invalidated_success[env_ids] = False


def _buffer_needs_reset(env, name: str, shape: tuple[int, ...], dtype: torch.dtype | None = None) -> bool:
    tensor = getattr(env, name, None)
    if not torch.is_tensor(tensor):
        return True
    target_device = torch.empty((), device=env.device).device
    if tensor.shape != shape or tensor.device != target_device:
        return True
    return dtype is not None and tensor.dtype != dtype


def _ensure_wall_brush_smoothness_buffers(env):
    count_shape = (env.scene.num_envs,)
    vector_shape = (env.scene.num_envs, 3)
    if (
        _buffer_needs_reset(env, "wall_brush_prev_tip_pos", vector_shape, torch.float32)
        or _buffer_needs_reset(env, "wall_brush_prev_tip_vel", vector_shape, torch.float32)
        or _buffer_needs_reset(env, "wall_brush_prev_tip_accel", vector_shape, torch.float32)
        or _buffer_needs_reset(env, "wall_brush_tip_history_count", count_shape, torch.long)
    ):
        env.wall_brush_prev_tip_pos = torch.zeros(vector_shape, device=env.device, dtype=torch.float32)
        env.wall_brush_prev_tip_vel = torch.zeros(vector_shape, device=env.device, dtype=torch.float32)
        env.wall_brush_prev_tip_accel = torch.zeros(vector_shape, device=env.device, dtype=torch.float32)
        env.wall_brush_tip_history_count = torch.zeros(count_shape, device=env.device, dtype=torch.long)
    action = getattr(getattr(env, "action_manager", None), "action", None)
    if action is not None and (
        _buffer_needs_reset(env, "wall_brush_prev_action_delta", tuple(action.shape), action.dtype)
        or _buffer_needs_reset(env, "wall_brush_action_delta_valid", count_shape, torch.bool)
    ):
        env.wall_brush_prev_action_delta = torch.zeros_like(action)
        env.wall_brush_action_delta_valid = torch.zeros(count_shape, device=env.device, dtype=torch.bool)


def reset_wall_brush_smoothness_buffers(env, env_ids: torch.Tensor):
    _ensure_wall_brush_smoothness_buffers(env)
    env.wall_brush_prev_tip_pos[env_ids] = 0.0
    env.wall_brush_prev_tip_vel[env_ids] = 0.0
    env.wall_brush_prev_tip_accel[env_ids] = 0.0
    env.wall_brush_tip_history_count[env_ids] = 0
    if hasattr(env, "wall_brush_prev_action_delta"):
        env.wall_brush_prev_action_delta[env_ids] = 0.0
    if hasattr(env, "wall_brush_action_delta_valid"):
        env.wall_brush_action_delta_valid[env_ids] = False


def _body_id(asset: Articulation, asset_cfg: SceneEntityCfg) -> int:
    if isinstance(asset_cfg.body_ids, slice):
        return asset.body_names.index(BRUSH_LINK)
    if isinstance(asset_cfg.body_ids, torch.Tensor):
        return int(asset_cfg.body_ids.flatten()[0].item())
    if isinstance(asset_cfg.body_ids, (list, tuple)):
        return int(asset_cfg.body_ids[0])
    return int(asset_cfg.body_ids)


def _body_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    if isinstance(asset_cfg.body_ids, slice):
        return torch.arange(len(asset.body_names), device=asset.data.root_pos_w.device, dtype=torch.long)
    if isinstance(asset_cfg.body_ids, torch.Tensor):
        return asset_cfg.body_ids.flatten().to(device=asset.data.root_pos_w.device, dtype=torch.long)
    if isinstance(asset_cfg.body_ids, (list, tuple)):
        return torch.as_tensor(asset_cfg.body_ids, device=asset.data.root_pos_w.device, dtype=torch.long)
    return torch.as_tensor([asset_cfg.body_ids], device=asset.data.root_pos_w.device, dtype=torch.long)


def _body_ids_from_names(asset: Articulation, names: list[str], device) -> torch.Tensor:
    ids = [asset.body_names.index(name) for name in names if name in asset.body_names]
    return torch.as_tensor(ids, device=device, dtype=torch.long)


def _sample_curriculum_motion_ids(env, env_ids: torch.Tensor) -> torch.Tensor:
    total_motions = int(env.total_motions)
    if total_motions == len(WALL_BRUSH_CURRICULUM_PRIOR_WEIGHTS):
        weights = torch.tensor(WALL_BRUSH_CURRICULUM_PRIOR_WEIGHTS, device=env.device, dtype=torch.float32)
        return torch.multinomial(weights, len(env_ids), replacement=True)
    return torch.randint(0, total_motions, (len(env_ids),), device=env.device)


def _sample_uniform_motion_ids(env, env_ids: torch.Tensor) -> torch.Tensor:
    return torch.randint(0, int(env.total_motions), (len(env_ids),), device=env.device)


def reset_joints_for_wall_brush_curriculum(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    if _has_motion_lib(env):
        env.motion_ids[env_ids] = _sample_curriculum_motion_ids(env, env_ids)
        env.start_motion_times[env_ids] = 0.0
        motion_times = env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)[env_ids]
        motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)[env_ids]
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        joint_ids = torch.tensor(asset_cfg.joint_ids, device=env.device)
        joint_pos[:, joint_ids] = motion_res["dof_pos"]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_joints_for_wall_brush_stance(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    if _has_motion_lib(env):
        env.motion_ids[env_ids] = _sample_uniform_motion_ids(env, env_ids)
        env.start_motion_times[env_ids] = 0.0
        motion_times = env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)[env_ids]
        motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)[env_ids]
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        joint_ids = torch.tensor(asset_cfg.joint_ids, device=env.device)
        joint_pos[:, joint_ids] = motion_res["dof_pos"]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_root_state_for_wall_brush_stance(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset_z: float = 0.15,
):
    asset: Articulation = env.scene[asset_cfg.name]
    if _has_motion_lib(env):
        motion_times = env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)[env_ids]
        motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)[env_ids]
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        positions = motion_res["root_pos"] + env.scene.env_origins[env_ids]
    else:
        positions = asset.data.default_root_state[env_ids, :3].clone() + env.scene.env_origins[env_ids]
    positions[:, 2] += offset_z
    orientations = torch.zeros((len(env_ids), 4), device=env.device)
    orientations[:, 0] = 1.0
    velocities = torch.zeros((len(env_ids), 6), device=env.device)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def _brush_tip_pos(env, asset_cfg: SceneEntityCfg, motion_res: dict) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    link_pos = asset.data.body_state_w[:, _body_id(asset, asset_cfg), :3].clone() - env.scene.env_origins
    return link_pos


def _body_wall_clearance(env, asset_cfg: SceneEntityCfg, motion_res: dict) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _body_ids(asset, asset_cfg)
    body_pos = asset.data.body_state_w[:, ids, :3] - env.scene.env_origins.unsqueeze(1)
    wall_x = motion_res["wall_mid"][:, 0].unsqueeze(1)
    return wall_x - body_pos[:, :, 0]


def _wall_contact_force(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    near_wall_margin: float,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return torch.zeros(env.num_envs, device=env.device)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.data.net_forces_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_state_w[:, asset_cfg.body_ids, :3] - env.scene.env_origins.unsqueeze(1)
    wall_x = motion_res["wall_mid"][:, 0].unsqueeze(1)
    near_wall = (wall_x - body_pos[:, :, 0]) < near_wall_margin
    wall_normal_force = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 0])
    force = torch.where(near_wall, wall_normal_force, torch.zeros_like(wall_normal_force))
    return torch.amax(force, dim=1)


def _self_collision_proxy_stats(
    env,
    asset_cfg: SceneEntityCfg,
    pair_groups: list[tuple[str, list[str], list[str], float]],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[asset_cfg.name]
    device = env.device
    body_pos = asset.data.body_state_w[:, :, :3]
    violations = []
    margins = []
    for _, group_a, group_b, min_distance in pair_groups:
        ids_a = _body_ids_from_names(asset, group_a, device)
        ids_b = _body_ids_from_names(asset, group_b, device)
        if ids_a.numel() == 0 or ids_b.numel() == 0:
            continue
        delta = body_pos[:, ids_a, None, :] - body_pos[:, None, ids_b, :]
        pair_dist = torch.norm(delta, dim=-1).reshape(env.scene.num_envs, -1)
        min_dist = torch.min(pair_dist, dim=1).values
        margins.append(min_dist - min_distance)
        violations.append(((min_distance - min_dist).clamp_min(0.0) / max(scale, 1e-6)) ** 2)
    if not violations:
        return (
            torch.zeros(env.scene.num_envs, device=device),
            torch.full((env.scene.num_envs,), float("inf"), device=device),
        )
    return torch.max(torch.stack(violations, dim=1), dim=1).values, torch.min(torch.stack(margins, dim=1), dim=1).values


def brush_tip_reference_tracking(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    std: float = 0.08,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    err = torch.norm(tip - motion_res["brush_tip_pos"], dim=1)
    return torch.exp(-(err / std) ** 2)


def brush_tip_wall_contact(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    broad_std: float = 0.06,
    fine_std: float = 0.02,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    wall_x = motion_res["wall_mid"][:, 0]
    err = torch.abs(tip[:, 0] - wall_x)
    active = motion_res["stroke_active"].float()
    broad = torch.exp(-((err / broad_std) ** 2))
    fine = torch.exp(-((err / fine_std) ** 2))
    return (0.6 * broad + 0.4 * fine) * active


def brush_tip_wall_band_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    target_band: float = 0.03,
    scale: float = 0.06,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    wall_x = motion_res["wall_mid"][:, 0]
    err = torch.abs(tip[:, 0] - wall_x)
    active = motion_res["stroke_active"].float()
    return ((err - target_band).clamp_min(0.0) / scale) ** 2 * active


def brush_tip_row_accuracy(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    broad_std: float = 0.18,
    fine_std: float = 0.04,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    err = torch.norm(tip[:, 1:3] - motion_res["brush_tip_pos"][:, 1:3], dim=1)
    active = motion_res["stroke_active"].float()
    broad = torch.exp(-((err / broad_std) ** 2))
    fine = torch.exp(-((err / fine_std) ** 2))
    return (0.4 * broad + 0.6 * fine) * active


def brush_tip_row_band_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    target_band: float = 0.05,
    scale: float = 0.12,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    err = torch.norm(tip[:, 1:3] - motion_res["brush_tip_pos"][:, 1:3], dim=1)
    active = motion_res["stroke_active"].float()
    return ((err - target_band).clamp_min(0.0) / scale) ** 2 * active


def brush_tip_hard_prior_row_band_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    target_band: float = 0.045,
    scale: float = 0.08,
    row_z_threshold: float = 0.94,
    wall_distance_threshold: float = 0.435,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    err = torch.norm(tip[:, 1:3] - motion_res["brush_tip_pos"][:, 1:3], dim=1)
    wall_mid = motion_res["wall_mid"]
    root_pos = motion_res["root_pos"]
    hard_prior = (wall_mid[:, 2] <= row_z_threshold) | (
        (wall_mid[:, 0] - root_pos[:, 0]) <= wall_distance_threshold
    )
    active = motion_res["stroke_active"].float() * hard_prior.float()
    return ((err - target_band).clamp_min(0.0) / scale) ** 2 * active


def brush_tip_line_progress(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    broad_std: float = 0.22,
    fine_std: float = 0.06,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    start = motion_res["wall_start"]
    end = motion_res["wall_end"]
    line = end - start
    denom = torch.sum(line * line, dim=1).clamp_min(1e-6)
    tip_phase = torch.sum((tip - start) * line, dim=1) / denom
    ref_phase = torch.sum((motion_res["brush_tip_pos"] - start) * line, dim=1) / denom
    err = torch.clamp(tip_phase, 0.0, 1.0) - torch.clamp(ref_phase, 0.0, 1.0)
    active = motion_res["stroke_active"].float()
    err = torch.abs(err)
    broad = torch.exp(-((err / broad_std) ** 2))
    fine = torch.exp(-((err / fine_std) ** 2))
    return (0.4 * broad + 0.6 * fine) * active


def brush_tip_phase_lag_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    margin: float = 0.05,
    scale: float = 0.25,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    start = motion_res["wall_start"]
    end = motion_res["wall_end"]
    line = end - start
    denom = torch.sum(line * line, dim=1).clamp_min(1e-6)
    tip_phase = torch.sum((tip - start) * line, dim=1) / denom
    ref_phase = torch.sum((motion_res["brush_tip_pos"] - start) * line, dim=1) / denom
    lag = (torch.clamp(ref_phase, 0.0, 1.0) - torch.clamp(tip_phase, 0.0, 1.0) - margin).clamp_min(0.0)
    return ((lag / scale) ** 2) * motion_res["stroke_active"].float()


def brush_tip_phase_band_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    target_band: float = 0.08,
    scale: float = 0.18,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    start = motion_res["wall_start"]
    end = motion_res["wall_end"]
    line = end - start
    denom = torch.sum(line * line, dim=1).clamp_min(1e-6)
    tip_phase = torch.sum((tip - start) * line, dim=1) / denom
    ref_phase = torch.sum((motion_res["brush_tip_pos"] - start) * line, dim=1) / denom
    err = torch.abs(torch.clamp(tip_phase, 0.0, 1.0) - torch.clamp(ref_phase, 0.0, 1.0))
    return ((err - target_band).clamp_min(0.0) / scale) ** 2 * motion_res["stroke_active"].float()


def brush_tip_late_progress(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    late_phase: float = 0.65,
    required_next_anchor: int | None = None,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    start = motion_res["wall_start"]
    end = motion_res["wall_end"]
    line = end - start
    denom = torch.sum(line * line, dim=1).clamp_min(1e-6)
    tip_phase = torch.sum((tip - start) * line, dim=1) / denom
    ref_phase = torch.sum((motion_res["brush_tip_pos"] - start) * line, dim=1) / denom
    late = (ref_phase >= late_phase).float() * motion_res["stroke_active"].float()
    if required_next_anchor is not None:
        _ensure_wall_brush_success_buffers(env)
        late = late * (env.wall_brush_next_anchor >= int(required_next_anchor)).float()
    return torch.clamp(tip_phase, 0.0, 1.0) * late


_brush_tip_late_progress_func = brush_tip_late_progress


def brush_tip_anchor_approach_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    max_step_reward: float = 0.08,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    _ensure_wall_brush_success_buffers(env)

    active = motion_res["stroke_active"].bool()
    reset_boundary = getattr(env, "reset_terminated", torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)).bool()
    if torch.any(reset_boundary):
        env.wall_brush_prev_anchor_dist[reset_boundary] = float("inf")

    update_mask = active & (~reset_boundary) & (~env.wall_brush_invalidated_success)
    reward = torch.zeros(env.scene.num_envs, device=env.device)
    if not torch.any(update_mask):
        return reward

    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    anchors = torch.stack([motion_res["wall_start"], motion_res["wall_mid"], motion_res["wall_end"]], dim=1)
    next_anchor = env.wall_brush_next_anchor.clamp(max=2)
    expected_anchor = anchors[torch.arange(env.scene.num_envs, device=env.device), next_anchor]
    wall_x = motion_res["wall_mid"][:, 0]
    anchor_delta = torch.cat([(tip[:, 0:1] - wall_x.unsqueeze(1)), tip[:, 1:3] - expected_anchor[:, 1:3]], dim=1)
    dist = torch.norm(anchor_delta, dim=1)

    prev_dist = env.wall_brush_prev_anchor_dist
    finite_prev = torch.isfinite(prev_dist)
    progress = torch.where(finite_prev, prev_dist - dist, torch.zeros_like(dist))
    reward[update_mask] = progress[update_mask].clamp(min=-max_step_reward, max=max_step_reward)
    env.wall_brush_prev_anchor_dist[update_mask] = dist[update_mask]
    return reward


def body_wall_clearance_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=NONBRUSH_LINKS),
    min_clearance: float = 0.08,
    active_only: bool = True,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    clearance = _body_wall_clearance(env, asset_cfg, motion_res)
    violation = ((min_clearance - clearance).clamp_min(0.0) / max(min_clearance, 1e-6)) ** 2
    penalty = torch.max(violation, dim=1).values
    if active_only:
        penalty = penalty * motion_res["stroke_active"].float()
    return penalty


def wall_contact_force_violation(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
    threshold: float = 1.0,
    near_wall_margin: float = 0.22,
) -> torch.Tensor:
    force = _wall_contact_force(env, sensor_cfg, asset_cfg, near_wall_margin)
    return torch.clamp(force / max(threshold, 1e-6), max=10.0)


def wall_contact_force_above_threshold(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
    threshold: float = 1.0,
    near_wall_margin: float = 0.22,
) -> torch.Tensor:
    return _wall_contact_force(env, sensor_cfg, asset_cfg, near_wall_margin) > threshold


def self_collision_proxy_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pair_groups: list[tuple[str, list[str], list[str], float]] = SELF_COLLISION_PAIR_GROUPS,
    scale: float = 0.04,
) -> torch.Tensor:
    violation, _ = _self_collision_proxy_stats(env, asset_cfg, pair_groups, scale)
    return violation


def self_collision_proxy_below_threshold(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pair_groups: list[tuple[str, list[str], list[str], float]] = SELF_COLLISION_PAIR_GROUPS,
    scale: float = 0.04,
    tolerance: float = 0.01,
) -> torch.Tensor:
    _, margin = _self_collision_proxy_stats(env, asset_cfg, pair_groups, scale)
    return margin < -abs(tolerance)


def brush_tip_anchor_milestones(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    contact_band: float = 0.03,
    row_band: float = 0.05,
    anchor_radius: float = 0.05,
    contact_ratio: float = 0.60,
    row_ratio: float = 0.60,
    combined_ratio: float = 0.50,
    coverage: float = 0.70,
    anchor_bonus: float = 1.0,
    all_anchor_bonus: float = 3.0,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    _ensure_wall_brush_success_buffers(env)

    active = motion_res["stroke_active"].bool()
    reset_boundary = getattr(env, "reset_terminated", torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)).bool()
    canceled = reset_boundary & (env.wall_brush_pending_success | env.wall_brush_countable_success)
    if torch.any(canceled):
        env.n_successes[canceled] = 0
    if torch.any(reset_boundary):
        env.wall_brush_pending_success[reset_boundary] = False
        env.wall_brush_countable_success[reset_boundary] = False
        env.wall_brush_invalidated_success[reset_boundary] = True

    update_mask = active & (~reset_boundary) & (~env.wall_brush_invalidated_success)
    reward = torch.zeros(env.scene.num_envs, device=env.device)
    if not torch.any(update_mask):
        return reward

    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    start = motion_res["wall_start"]
    mid = motion_res["wall_mid"]
    end = motion_res["wall_end"]
    wall_x = mid[:, 0]
    legal_contact = torch.abs(tip[:, 0] - wall_x) <= contact_band

    line_yz = end[:, 1:3] - start[:, 1:3]
    denom = torch.sum(line_yz * line_yz, dim=1).clamp_min(1e-6)
    phase = torch.sum((tip[:, 1:3] - start[:, 1:3]) * line_yz, dim=1) / denom
    phase = torch.clamp(phase, 0.0, 1.0)
    row_projection = start[:, 1:3] + line_yz * phase.unsqueeze(1)
    row_dist = torch.norm(tip[:, 1:3] - row_projection, dim=1)
    row_valid = row_dist <= row_band
    combined_valid = legal_contact & row_valid

    env.wall_brush_active_steps[update_mask] += 1.0
    env.wall_brush_contact_steps[update_mask & legal_contact] += 1.0
    env.wall_brush_row_steps[update_mask & row_valid] += 1.0
    env.wall_brush_combined_steps[update_mask & combined_valid] += 1.0
    combined_update = update_mask & combined_valid
    env.wall_brush_min_phase[combined_update] = torch.minimum(env.wall_brush_min_phase[combined_update], phase[combined_update])
    env.wall_brush_max_phase[combined_update] = torch.maximum(env.wall_brush_max_phase[combined_update], phase[combined_update])

    anchors = torch.stack([start, mid, end], dim=1)
    next_anchor = env.wall_brush_next_anchor.clamp(max=2)
    expected_anchor = anchors[torch.arange(env.scene.num_envs, device=env.device), next_anchor]
    anchor_dist = torch.norm(tip[:, 1:3] - expected_anchor[:, 1:3], dim=1)
    anchor_hit = update_mask & legal_contact & (env.wall_brush_next_anchor < 3) & (anchor_dist <= anchor_radius)
    if torch.any(anchor_hit):
        env.wall_brush_next_anchor[anchor_hit] += 1
        reward[anchor_hit] += anchor_bonus
        all_hit = anchor_hit & (env.wall_brush_next_anchor >= 3)
        reward[all_hit] += all_anchor_bonus

    active_steps = env.wall_brush_active_steps.clamp_min(1.0)
    phase_coverage = (env.wall_brush_max_phase - env.wall_brush_min_phase).clamp_min(0.0)
    success = (
        (env.wall_brush_next_anchor >= 3)
        & (env.wall_brush_contact_steps / active_steps >= contact_ratio)
        & (env.wall_brush_row_steps / active_steps >= row_ratio)
        & (env.wall_brush_combined_steps / active_steps >= combined_ratio)
        & (phase_coverage >= coverage)
        & (~env.wall_brush_invalidated_success)
    )
    new_success = success & (~env.wall_brush_countable_success)
    if torch.any(new_success):
        env.n_successes[new_success] += 1
    env.wall_brush_pending_success[success] = True
    env.wall_brush_countable_success[success] = True
    return reward


def foot_reference_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=FOOT_LINKS),
    std: float = 0.05,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _body_ids(asset, asset_cfg)
    foot_pos = asset.data.body_state_w[:, ids, :3] - env.scene.env_origins.unsqueeze(1)
    ref_pos = motion_res["global_keypts"][:, FOOT_KEYPOINT_IDS, :]
    err = torch.norm(foot_pos - ref_pos, dim=2)
    return torch.max((err / std) ** 2, dim=1).values


def stance_joint_reference_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
    joint_mask: list[float] = JOINTS_MASK,
) -> torch.Tensor:
    motion_res = _stance_motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    asset: Articulation = env.scene[asset_cfg.name]
    ref_joint_pos = motion_res["dof_pos"]
    mask = torch.tensor(joint_mask, device=env.device, dtype=ref_joint_pos.dtype).unsqueeze(0)
    err = (asset.data.joint_pos[:, asset_cfg.joint_ids] - ref_joint_pos) * mask
    return torch.sum(torch.abs(err), dim=1)


def action_joint_reference_violation(
    env,
    action_name: str = "joint_pos",
    joint_mask: list[float] = RIGHT_ARM_JOINTS_MASK,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    action_term = env.action_manager.get_term(action_name)
    ref_joint_pos = motion_res["dof_pos"]
    mask = torch.tensor(joint_mask, device=env.device, dtype=ref_joint_pos.dtype).unsqueeze(0)
    err = (action_term.processed_actions - ref_joint_pos) * mask
    return torch.sum(torch.abs(err), dim=1)


def action_accel_l2(env) -> torch.Tensor:
    _ensure_wall_brush_smoothness_buffers(env)
    action = env.action_manager.action
    delta = action - env.action_manager.prev_action
    accel = delta - env.wall_brush_prev_action_delta
    penalty = torch.sum(torch.square(accel), dim=1)
    penalty = torch.where(env.wall_brush_action_delta_valid, penalty, torch.zeros_like(penalty))
    env.wall_brush_prev_action_delta[:] = delta
    env.wall_brush_action_delta_valid[:] = True
    return penalty


def brush_tip_smoothness_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    accel_scale: float = 20.0,
    jerk_scale: float = 600.0,
    jerk_mix: float = 0.35,
    active_only: bool = False,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    _ensure_wall_brush_smoothness_buffers(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    dt = max(float(getattr(env, "step_dt", 0.02)), 1e-6)
    count = env.wall_brush_tip_history_count
    has_prev_pos = count >= 1
    has_prev_vel = count >= 2
    has_prev_accel = count >= 3

    vel = (tip - env.wall_brush_prev_tip_pos) / dt
    accel = (vel - env.wall_brush_prev_tip_vel) / dt
    jerk = (accel - env.wall_brush_prev_tip_accel) / dt
    zero = torch.zeros_like(tip)
    accel = torch.where(has_prev_vel.unsqueeze(1), accel, zero)
    jerk = torch.where(has_prev_accel.unsqueeze(1), jerk, zero)

    accel_penalty = torch.sum(torch.square(accel / max(accel_scale, 1e-6)), dim=1)
    jerk_penalty = torch.sum(torch.square(jerk / max(jerk_scale, 1e-6)), dim=1)
    penalty = accel_penalty + jerk_mix * jerk_penalty
    if active_only:
        penalty = penalty * motion_res["stroke_active"].float()

    env.wall_brush_prev_tip_pos[:] = tip
    env.wall_brush_prev_tip_vel[:] = torch.where(has_prev_pos.unsqueeze(1), vel, zero)
    env.wall_brush_prev_tip_accel[:] = torch.where(has_prev_vel.unsqueeze(1), accel, zero)
    env.wall_brush_tip_history_count[:] = torch.clamp(count + 1, max=3)
    return penalty


def stance_root_position_error(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    motion_res = _stance_motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    return torch.norm(root_pos - motion_res["root_pos"], dim=1)


def stance_root_orientation_error(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    motion_res = _stance_motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.quat_error_magnitude(motion_res["root_rot"], asset.data.root_quat_w)


def root_upright_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    angle_threshold_deg: float = 12.0,
    scale_deg: float = 18.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_rot = math_utils.matrix_from_quat(math_utils.quat_unique(asset.data.root_quat_w))
    z_axis = root_rot[:, :, 2]
    angle = torch.acos(z_axis[:, 2].clamp(-1.0, 1.0))
    threshold = torch.deg2rad(torch.tensor(angle_threshold_deg, device=env.device))
    scale = torch.deg2rad(torch.tensor(scale_deg, device=env.device)).clamp_min(1e-6)
    return ((angle - threshold).clamp_min(0.0) / scale) ** 2


def root_height_floor_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.72,
    scale: float = 0.15,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    return ((min_height - root_pos[:, 2]).clamp_min(0.0) / max(scale, 1e-6)) ** 2


def foot_stance_reference_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=FOOT_LINKS),
    std: float = 0.035,
) -> torch.Tensor:
    motion_res = _stance_motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _body_ids(asset, asset_cfg)
    foot_pos = asset.data.body_state_w[:, ids, :3] - env.scene.env_origins.unsqueeze(1)
    ref_pos = motion_res["global_keypts"][:, FOOT_KEYPOINT_IDS, :]
    err = torch.norm(foot_pos - ref_pos, dim=2)
    return torch.max((err / std) ** 2, dim=1).values


def wall_facing_yaw_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw_threshold_deg: float = 20.0,
    active_only: bool = True,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    asset: Articulation = env.scene[asset_cfg.name]
    root_rot = math_utils.matrix_from_quat(math_utils.quat_unique(asset.data.root_quat_w))
    forward = root_rot[:, :, 0]
    forward_xy = forward[:, :2] / torch.norm(forward[:, :2], dim=1, keepdim=True).clamp_min(1e-6)
    wall_normal_xy = torch.zeros_like(forward_xy)
    wall_normal_xy[:, 0] = 1.0
    yaw = torch.acos(torch.sum(forward_xy * wall_normal_xy, dim=1).clamp(-1.0, 1.0))
    threshold = yaw_threshold_deg * torch.pi / 180.0
    penalty = ((yaw - threshold).clamp_min(0.0) / max(float(threshold), 1e-6)) ** 2
    if active_only:
        penalty = penalty * motion_res["stroke_active"].float()
    return penalty


def body_wall_clearance_below_threshold(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=TORSO_HEAD_LINKS),
    min_clearance: float = 0.08,
    active_only: bool = False,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)
    clearance = _body_wall_clearance(env, asset_cfg, motion_res)
    terminated = torch.min(clearance, dim=1).values < min_clearance
    if active_only:
        terminated = torch.logical_and(terminated, motion_res["stroke_active"].bool())
    return terminated


def wall_proxy_clearance_below_threshold(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
    min_clearance: float = 0.03,
    proxy_offsets: list[list[float]] = TORSO_HEAD_WALL_PROXY_OFFSETS,
    active_only: bool = False,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.bool)
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _body_id(asset, asset_cfg)
    body_pos = asset.data.body_state_w[:, body_id, :3] - env.scene.env_origins
    body_quat = math_utils.quat_unique(asset.data.body_state_w[:, body_id, 3:7])
    offsets = torch.tensor(proxy_offsets, device=env.device, dtype=body_pos.dtype)
    offsets = offsets.unsqueeze(0).expand(env.scene.num_envs, -1, -1)
    quat = body_quat.unsqueeze(1).expand(-1, offsets.shape[1], -1)
    rotated_offsets = math_utils.quat_apply(quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(
        env.scene.num_envs, offsets.shape[1], 3
    )
    proxy_pos = body_pos.unsqueeze(1) + rotated_offsets
    wall_x = motion_res["wall_mid"][:, 0].unsqueeze(1)
    terminated = torch.min(wall_x - proxy_pos[:, :, 0], dim=1).values < min_clearance
    if active_only:
        terminated = torch.logical_and(terminated, motion_res["stroke_active"].bool())
    return terminated


def wall_proxy_clearance_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
    min_clearance: float = 0.10,
    scale: float = 0.10,
    proxy_offsets: list[list[float]] = TORSO_HEAD_WALL_PROXY_OFFSETS,
    active_only: bool = False,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _body_id(asset, asset_cfg)
    body_pos = asset.data.body_state_w[:, body_id, :3] - env.scene.env_origins
    body_quat = math_utils.quat_unique(asset.data.body_state_w[:, body_id, 3:7])
    offsets = torch.tensor(proxy_offsets, device=env.device, dtype=body_pos.dtype)
    offsets = offsets.unsqueeze(0).expand(env.scene.num_envs, -1, -1)
    quat = body_quat.unsqueeze(1).expand(-1, offsets.shape[1], -1)
    rotated_offsets = math_utils.quat_apply(quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(
        env.scene.num_envs, offsets.shape[1], 3
    )
    proxy_pos = body_pos.unsqueeze(1) + rotated_offsets
    wall_x = motion_res["wall_mid"][:, 0].unsqueeze(1)
    clearance = wall_x - proxy_pos[:, :, 0]
    violation = ((min_clearance - clearance).clamp_min(0.0) / max(scale, 1e-6)) ** 2
    penalty = torch.max(violation, dim=1).values
    if active_only:
        penalty = penalty * motion_res["stroke_active"].float()
    return penalty


def wall_brush_target_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return torch.zeros(env.scene.num_envs, 10, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    root_rot = math_utils.quat_unique(asset.data.root_quat_w)
    root_rot_inv = math_utils.quat_conjugate(root_rot)
    points = torch.stack(
        [motion_res["brush_tip_pos"], motion_res["wall_start"], motion_res["wall_end"]],
        dim=1,
    )
    local = math_utils.quat_apply(
        root_rot_inv.unsqueeze(1).repeat_interleave(points.shape[1], dim=1),
        points - root_pos.unsqueeze(1),
    ).reshape(env.scene.num_envs, -1)
    return torch.cat([local, motion_res["stroke_active"].float().unsqueeze(1)], dim=1)


def brush_tip_target_error_obs(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
) -> torch.Tensor:
    """Current brush-tip error to the time-aligned virtual target in the robot frame."""

    motion_res = _motion_state(env)
    if motion_res is None:
        return torch.zeros(env.scene.num_envs, 3, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _body_id(asset, asset_cfg)
    root_rot_inv = math_utils.quat_conjugate(math_utils.quat_unique(asset.data.root_quat_w))
    tip = asset.data.body_state_w[:, body_id, :3] - env.scene.env_origins
    delta = motion_res["brush_tip_pos"] - tip
    return math_utils.quat_apply(root_rot_inv, delta)


@configclass
class EventCfg(EventCfgBase):
    reset_success_buffers = EventTerm(
        func=reset_wall_brush_success_buffers,
        mode="reset",
    )
    reset_base = EventTerm(
        func=reset_root_state_for_motion,
        mode="reset",
        params={"offset_z": 0.15},
    )


@configclass
class G1Rewards(G1RewardsBase):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-6000.0)
    alive_reward = RewTerm(func=mdp.is_alive, weight=6.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-16.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_parallel_to_ground = RewTerm(
        func=mdp.feet_parallel_to_ground,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )
    joint_deviation_ref = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-0.85,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "joint_mask": JOINTS_MASK},
    )
    keypts_deviation_ref = RewTerm(
        func=keypts_deviation_ref_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "keypts_mask": KEYPTS_MASK},
    )
    position_tracking_error = RewTerm(
        func=position_tracking_error,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    orientation_tracking_error = RewTerm(
        func=orientation_tracking_error,
        weight=-0.60,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-250.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    brush_tip_reference = RewTerm(
        func=brush_tip_reference_tracking,
        weight=20.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_contact = RewTerm(
        func=brush_tip_wall_contact,
        weight=24.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_contact_band = RewTerm(
        func=brush_tip_wall_band_violation,
        weight=-12.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=36.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_row_band = RewTerm(
        func=brush_tip_row_band_violation,
        weight=-30.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_progress = RewTerm(
        func=brush_tip_line_progress,
        weight=42.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_phase_band = RewTerm(
        func=brush_tip_phase_band_violation,
        weight=-28.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.08, "scale": 0.18},
    )
    brush_tip_phase_lag = RewTerm(
        func=brush_tip_phase_lag_violation,
        weight=-28.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_late_progress = RewTerm(
        func=brush_tip_late_progress,
        weight=24.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_endpoint_progress = RewTerm(
        func=_brush_tip_late_progress_func,
        weight=18.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "late_phase": 0.82},
    )
    brush_tip_anchor_milestones = RewTerm(
        func=brush_tip_anchor_milestones,
        weight=8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
            "contact_band": 0.03,
            "row_band": 0.05,
            "anchor_radius": 0.05,
            "contact_ratio": 0.60,
            "row_ratio": 0.60,
            "combined_ratio": 0.50,
            "coverage": 0.70,
            "anchor_bonus": 1.0,
            "all_anchor_bonus": 3.0,
        },
    )
    stance_upright = RewTerm(
        func=root_upright_violation,
        weight=-80.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "angle_threshold_deg": 6.0, "scale_deg": 12.0},
    )
    torso_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=TORSO_HEAD_LINKS),
            "min_clearance": 0.15,
            "active_only": False,
        },
    )
    torso_head_wall_proxy_clearance = RewTerm(
        func=wall_proxy_clearance_violation,
        weight=-32.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "min_clearance": 0.15,
            "scale": 0.12,
            "active_only": False,
        },
    )
    nonbrush_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-30.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=NONBRUSH_LINKS),
            "min_clearance": 0.14,
            "active_only": False,
        },
    )
    right_hand_wall_clearance = None
    nonbrush_wall_contact = RewTerm(
        func=wall_contact_force_violation,
        weight=-120.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "threshold": 0.5,
            "near_wall_margin": 0.28,
        },
    )
    left_arm_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-24.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_ARM_LINKS),
            "min_clearance": 0.14,
            "active_only": False,
        },
    )
    foot_reference = RewTerm(
        func=foot_reference_violation,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINKS)},
    )
    wall_facing_yaw = RewTerm(
        func=wall_facing_yaw_violation,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "yaw_threshold_deg": 20.0, "active_only": False},
    )


@configclass
class G1WallBrushCurriculumRewards(G1Rewards):
    brush_tip_reference = RewTerm(
        func=brush_tip_reference_tracking,
        weight=24.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=44.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_phase_lag = RewTerm(
        func=brush_tip_phase_lag_violation,
        weight=-18.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "scale": 0.28},
    )
    brush_tip_late_progress = RewTerm(
        func=_brush_tip_late_progress_func,
        weight=14.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "late_phase": 0.65},
    )
    brush_tip_endpoint_progress = RewTerm(
        func=_brush_tip_late_progress_func,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "late_phase": 0.82},
    )
    nonbrush_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-24.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=NONBRUSH_LINKS),
            "min_clearance": 0.14,
            "active_only": False,
        },
    )
    nonbrush_wall_contact = RewTerm(
        func=wall_contact_force_violation,
        weight=-120.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "threshold": 0.5,
            "near_wall_margin": 0.28,
        },
    )
    right_hand_wall_clearance = None
    left_arm_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_ARM_LINKS),
            "min_clearance": 0.11,
            "active_only": False,
        },
    )
    foot_reference = RewTerm(
        func=foot_reference_violation,
        weight=-2.4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINKS), "std": 0.045},
    )


@configclass
class G1WallBrushImitationRewards(G1RewardsBase):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-6000.0)
    alive_reward = RewTerm(func=mdp.is_alive, weight=4.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-8.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_parallel_to_ground = RewTerm(
        func=mdp.feet_parallel_to_ground,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )
    joint_deviation_ref = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "joint_mask": JOINTS_MASK},
    )
    lower_body_reference = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": LOWER_BODY_JOINTS_MASK,
        },
    )
    right_arm_reference = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": RIGHT_ARM_JOINTS_MASK,
        },
    )
    keypts_deviation_ref = RewTerm(
        func=keypts_deviation_ref_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "keypts_mask": KEYPTS_MASK},
    )
    position_tracking_error = RewTerm(
        func=position_tracking_error,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    orientation_tracking_error = RewTerm(
        func=orientation_tracking_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    brush_tip_reference = RewTerm(
        func=brush_tip_reference_tracking,
        weight=30.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "std": 0.10},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=14.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.22, "fine_std": 0.07},
    )
    brush_tip_progress = RewTerm(
        func=brush_tip_line_progress,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.28, "fine_std": 0.10},
    )
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-400.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stance_upright = RewTerm(
        func=root_upright_violation,
        weight=-120.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "angle_threshold_deg": 5.0, "scale_deg": 12.0},
    )
    torso_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-16.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=TORSO_HEAD_LINKS),
            "min_clearance": 0.17,
            "active_only": False,
        },
    )
    torso_head_wall_proxy_clearance = RewTerm(
        func=wall_proxy_clearance_violation,
        weight=-32.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "min_clearance": 0.15,
            "scale": 0.12,
            "active_only": False,
        },
    )
    nonbrush_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-16.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=NONBRUSH_LINKS),
            "min_clearance": 0.12,
            "active_only": False,
        },
    )
    nonbrush_wall_contact = RewTerm(
        func=wall_contact_force_violation,
        weight=-120.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "threshold": 0.5,
            "near_wall_margin": 0.28,
        },
    )
    left_arm_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_ARM_LINKS),
            "min_clearance": 0.12,
            "active_only": False,
        },
    )
    foot_reference = RewTerm(
        func=foot_reference_violation,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINKS), "std": 0.045},
    )
    wall_facing_yaw = RewTerm(
        func=wall_facing_yaw_violation,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "yaw_threshold_deg": 18.0, "active_only": False},
    )


@configclass
class G1WallBrushReachRewards(G1WallBrushImitationRewards):
    """Conservative reach stage: keep standing while forcing the right arm toward the motion prior."""

    alive_reward = RewTerm(func=mdp.is_alive, weight=6.0)
    joint_deviation_ref = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "joint_mask": JOINTS_MASK},
    )
    lower_body_reference = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": LOWER_BODY_JOINTS_MASK,
        },
    )
    right_arm_reference = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": RIGHT_ARM_JOINTS_MASK,
        },
    )
    keypts_deviation_ref = RewTerm(
        func=keypts_deviation_ref_l2,
        weight=-0.7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "keypts_mask": KEYPTS_MASK},
    )
    brush_tip_reference = RewTerm(
        func=brush_tip_reference_tracking,
        weight=80.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "std": 0.08},
    )
    brush_tip_contact = RewTerm(
        func=brush_tip_wall_contact,
        weight=18.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.08, "fine_std": 0.025},
    )
    brush_tip_contact_band = RewTerm(
        func=brush_tip_wall_band_violation,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.035, "scale": 0.08},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=20.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.22, "fine_std": 0.06},
    )
    brush_tip_row_band = RewTerm(
        func=brush_tip_row_band_violation,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.06, "scale": 0.14},
    )
    brush_tip_progress = RewTerm(
        func=brush_tip_line_progress,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.30, "fine_std": 0.12},
    )
    brush_tip_phase_band = RewTerm(
        func=brush_tip_phase_band_violation,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.12, "scale": 0.25},
    )
    brush_tip_phase_lag = None
    brush_tip_late_progress = None
    brush_tip_endpoint_progress = None
    brush_tip_anchor_milestones = None
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-500.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_reference = RewTerm(
        func=foot_reference_violation,
        weight=-8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINKS), "std": 0.04},
    )


@configclass
class G1WallBrushActionPriorRewards(G1WallBrushReachRewards):
    """Action-prior stage: make the policy command the reference right arm before long RL refinement."""

    right_arm_action_reference = RewTerm(
        func=action_joint_reference_violation,
        weight=-22.0,
        params={"action_name": "joint_pos", "joint_mask": RIGHT_ARM_JOINTS_MASK},
    )
    lower_body_action_reference = RewTerm(
        func=action_joint_reference_violation,
        weight=-8.0,
        params={"action_name": "joint_pos", "joint_mask": LOWER_BODY_JOINTS_MASK},
    )
    brush_tip_contact = RewTerm(
        func=brush_tip_wall_contact,
        weight=42.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.10, "fine_std": 0.03},
    )
    brush_tip_contact_band = RewTerm(
        func=brush_tip_wall_band_violation,
        weight=-28.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.03, "scale": 0.06},
    )
    brush_tip_reference = RewTerm(
        func=brush_tip_reference_tracking,
        weight=70.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "std": 0.08},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=16.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.22, "fine_std": 0.06},
    )
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-700.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class G1WallBrushNoWallCollisionRewards(G1WallBrushActionPriorRewards):
    """DreamControl-style virtual wall: no physical/illegal wall-contact shaping."""

    torso_wall_clearance = None
    torso_head_wall_proxy_clearance = None
    nonbrush_wall_clearance = None
    nonbrush_wall_contact = None
    left_arm_wall_clearance = None


@configclass
class G1WallBrushDreamControlWarmstartRewards(G1WallBrushNoWallCollisionRewards):
    """Root-unlocked full-body warm start before enforcing tight wall-brush contact."""

    alive_reward = RewTerm(func=mdp.is_alive, weight=10.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-8000.0)
    lower_body_reference = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": LOWER_BODY_JOINTS_MASK,
        },
    )
    right_arm_reference = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-16.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": RIGHT_ARM_JOINTS_MASK,
        },
    )
    brush_tip_reference = RewTerm(
        func=brush_tip_reference_tracking,
        weight=45.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "std": 0.10},
    )
    brush_tip_contact = None
    brush_tip_contact_band = None
    brush_tip_phase_band = None
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-80.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class G1WallBrushDreamControlStandStillRowContactRewards(G1WallBrushDreamControlWarmstartRewards):
    """Stand_Still continuation stage that adds gentle wall contact while preserving warmstart stability."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-8000.0)
    alive_reward = RewTerm(func=mdp.is_alive, weight=12.0)
    right_arm_reference = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-18.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": RIGHT_ARM_JOINTS_MASK,
        },
    )
    brush_tip_contact = RewTerm(
        func=brush_tip_wall_contact,
        weight=8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.12, "fine_std": 0.04},
    )
    brush_tip_contact_band = RewTerm(
        func=brush_tip_wall_band_violation,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.04, "scale": 0.08},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=24.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.22, "fine_std": 0.06},
    )
    brush_tip_row_band = RewTerm(
        func=brush_tip_row_band_violation,
        weight=-14.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.06, "scale": 0.14},
    )
    brush_tip_progress = RewTerm(
        func=brush_tip_line_progress,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.30, "fine_std": 0.12},
    )
    stability_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class G1WallBrushDreamControlBalanceWarmstartRewards(G1WallBrushDreamControlWarmstartRewards):
    """Short-horizon standing curriculum with the same full-body action and observation contract."""

    alive_reward = RewTerm(func=mdp.is_alive, weight=16.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-8000.0)
    lower_body_reference = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": LOWER_BODY_JOINTS_MASK,
        },
    )
    right_arm_reference = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": RIGHT_ARM_JOINTS_MASK,
        },
    )
    brush_tip_reference = None
    brush_tip_contact = None
    brush_tip_contact_band = None
    brush_tip_row = None
    brush_tip_row_band = None
    brush_tip_progress = None
    brush_tip_phase_band = None
    right_arm_action_reference = None
    lower_body_action_reference = None
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-40.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class G1WallBrushDreamControlButtonPressAlignedRewards(G1RewardsBase):
    """ButtonPress-scale tracking and task rewards for root-unlocked wall brushing."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    alive_reward = RewTerm(func=mdp.is_alive, weight=1.0)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINKS),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINKS),
        },
    )
    feet_parallel_to_ground = RewTerm(
        func=mdp.feet_parallel_to_ground,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINKS)},
    )
    joint_deviation_ref = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": BUTTONPRESS_TRACKING_JOINTS_MASK,
        },
    )
    keypts_deviation_ref = RewTerm(
        func=keypts_deviation_ref_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "keypts_mask": BUTTONPRESS_TRACKING_KEYPTS_MASK,
        },
    )
    position_tracking_error = RewTerm(
        func=position_tracking_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    orientation_tracking_error = RewTerm(
        func=orientation_tracking_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    root_upright = RewTerm(
        func=root_upright_violation,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "angle_threshold_deg": 8.0, "scale_deg": 16.0},
    )
    root_height_floor = RewTerm(
        func=root_height_floor_violation,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.72, "scale": 0.15},
    )
    brush_tip_reference = RewTerm(
        func=brush_tip_reference_tracking,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "std": 0.12},
    )
    brush_tip_contact = RewTerm(
        func=brush_tip_wall_contact,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.10, "fine_std": 0.03},
    )
    brush_tip_contact_band = RewTerm(
        func=brush_tip_wall_band_violation,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.03, "scale": 0.06},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.24, "fine_std": 0.08},
    )
    brush_tip_progress = RewTerm(
        func=brush_tip_line_progress,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.30, "fine_std": 0.12},
    )
    brush_tip_anchor_milestones = RewTerm(
        func=brush_tip_anchor_milestones,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
            "contact_band": 0.03,
            "row_band": 0.05,
            "anchor_radius": 0.05,
            "contact_ratio": 0.60,
            "row_ratio": 0.60,
            "combined_ratio": 0.50,
            "coverage": 0.70,
            "anchor_bonus": 1.0,
            "all_anchor_bonus": 3.0,
        },
    )


@configclass
class G1WallBrushDreamControlButtonPressAlignedAntiJitterRewards(G1WallBrushDreamControlButtonPressAlignedRewards):
    """ButtonPressAligned reward variant that reduces jitter and kinematic collision proxies."""

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.018)
    action_accel_l2 = RewTerm(func=action_accel_l2, weight=-0.006)
    brush_tip_smoothness = RewTerm(
        func=brush_tip_smoothness_l2,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
            "accel_scale": 20.0,
            "jerk_scale": 600.0,
            "jerk_mix": 0.35,
        },
    )
    brush_tip_contact = RewTerm(
        func=brush_tip_wall_contact,
        weight=2.4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.12, "fine_std": 0.04},
    )
    brush_tip_contact_band = RewTerm(
        func=brush_tip_wall_band_violation,
        weight=-1.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.035, "scale": 0.08},
    )
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-60.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    nonbrush_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=NONBRUSH_LINKS, preserve_order=True),
            "min_clearance": 0.08,
            "active_only": False,
        },
    )
    right_hand_nonbrush_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_HAND_NONBRUSH_LINKS, preserve_order=True),
            "min_clearance": 0.025,
            "active_only": True,
        },
    )


@configclass
class G1WallBrushDreamControlButtonPressBalanceRewards(G1WallBrushDreamControlButtonPressAlignedRewards):
    """ButtonPress-scale survival curriculum that permits balance residuals before brush tracking."""

    joint_deviation_ref = None
    keypts_deviation_ref = None
    position_tracking_error = None
    orientation_tracking_error = None
    brush_tip_reference = None
    brush_tip_contact = None
    brush_tip_contact_band = None
    brush_tip_row = None
    brush_tip_progress = None
    brush_tip_anchor_milestones = None
    root_upright = RewTerm(
        func=root_upright_violation,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "angle_threshold_deg": 8.0, "scale_deg": 16.0},
    )
    root_height_floor = RewTerm(
        func=root_height_floor_violation,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.72, "scale": 0.15},
    )


class G1WallBrushUpperBodyRewards(G1RewardsBase):
    """Official UB-style fixed-base manipulation rewards with virtual wall targets."""

    joint_deviation_ref = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": UPPER_BODY_WITH_WAIST_JOINTS_MASK,
        },
    )
    keypts_deviation_ref = RewTerm(
        func=keypts_deviation_ref_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "keypts_mask": KEYPTS_MASK},
    )
    brush_tip_reference = RewTerm(
        func=brush_tip_reference_tracking,
        weight=70.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "std": 0.08},
    )
    brush_tip_contact = RewTerm(
        func=brush_tip_wall_contact,
        weight=42.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.10, "fine_std": 0.03},
    )
    brush_tip_contact_band = RewTerm(
        func=brush_tip_wall_band_violation,
        weight=-28.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.03, "scale": 0.06},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=16.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.22, "fine_std": 0.06},
    )
    brush_tip_phase_lag = RewTerm(
        func=brush_tip_phase_lag_violation,
        weight=-16.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_late_progress = RewTerm(
        func=_brush_tip_late_progress_func,
        weight=20.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_anchor_milestones = RewTerm(
        func=brush_tip_anchor_milestones,
        weight=8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
            "contact_band": 0.03,
            "row_band": 0.05,
            "anchor_radius": 0.05,
            "contact_ratio": 0.60,
            "row_ratio": 0.60,
            "combined_ratio": 0.50,
            "coverage": 0.70,
            "anchor_bonus": 1.0,
            "all_anchor_bonus": 3.0,
        },
    )
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-250.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


class G1WallBrushRightArmUpperBodyRewards(G1WallBrushUpperBodyRewards):
    """Fixed-base manipulation rewards that leave the non-functional arm at default."""

    joint_deviation_ref = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": RIGHT_ARM_WITH_WAIST_JOINTS_MASK,
        },
    )
    keypts_deviation_ref = RewTerm(
        func=keypts_deviation_ref_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "keypts_mask": RIGHT_ARM_WITH_TORSO_KEYPTS_MASK,
        },
    )


class G1WallBrushRightArmAnchorRewards(G1WallBrushRightArmUpperBodyRewards):
    """Right-arm rewards with ordered-anchor shaping aligned to the success definition."""

    brush_tip_contact = RewTerm(
        func=brush_tip_wall_contact,
        weight=34.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.10, "fine_std": 0.03},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=24.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.22, "fine_std": 0.06},
    )
    brush_tip_anchor_approach = RewTerm(
        func=brush_tip_anchor_approach_reward,
        weight=90.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "max_step_reward": 0.08},
    )
    brush_tip_late_progress = RewTerm(
        func=_brush_tip_late_progress_func,
        weight=36.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK])},
    )
    brush_tip_anchor_milestones = RewTerm(
        func=brush_tip_anchor_milestones,
        weight=40.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
            "contact_band": 0.03,
            "row_band": 0.05,
            "anchor_radius": 0.055,
            "contact_ratio": 0.60,
            "row_ratio": 0.60,
            "combined_ratio": 0.50,
            "coverage": 0.70,
            "anchor_bonus": 2.0,
            "all_anchor_bonus": 6.0,
        },
    )


class G1WallBrushResidualAnchorRegularizedRewards(G1WallBrushRightArmAnchorRewards):
    """Residual-anchor rewards with a small residual-action penalty to reduce policy drift."""

    residual_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.035)


class G1WallBrushDreamControlAgileBaseRewards(G1WallBrushResidualAnchorRegularizedRewards):
    """Right-arm wall-brush rewards for the frozen AGILE lower-body base experiment."""

    lower_body_action_reference = None
    right_arm_action_reference = None
    residual_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.02)


class G1WallBrushTargetedLowRowRewards(G1WallBrushResidualAnchorRegularizedRewards):
    """Difficulty-focused shaping for low/near brush rows without adding physical wall collision."""

    residual_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.012)
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=30.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.22, "fine_std": 0.055},
    )
    brush_tip_anchor_approach = RewTerm(
        func=brush_tip_anchor_approach_reward,
        weight=105.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "max_step_reward": 0.08},
    )
    brush_tip_hard_prior_row_band = RewTerm(
        func=brush_tip_hard_prior_row_band_violation,
        weight=-18.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
            "target_band": 0.045,
            "scale": 0.08,
            "row_z_threshold": 0.94,
            "wall_distance_threshold": 0.435,
        },
    )
    right_hand_leg_self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-650.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pair_groups": RIGHT_HAND_LEG_COLLISION_PAIR_GROUPS,
            "scale": 0.03,
        },
    )


class G1WallBrushStanceRewards(G1RewardsBase):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-6000.0)
    alive_reward = RewTerm(func=mdp.is_alive, weight=4.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_parallel_to_ground = RewTerm(
        func=mdp.feet_parallel_to_ground,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )
    stance_lower_body_reference = RewTerm(
        func=stance_joint_reference_violation,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": LOWER_BODY_JOINTS_MASK,
        },
    )
    stance_upper_body_reference = RewTerm(
        func=stance_joint_reference_violation,
        weight=-9.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
            "joint_mask": UPPER_BODY_JOINTS_MASK,
        },
    )
    stance_root_position = RewTerm(
        func=stance_root_position_error,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stance_root_orientation = RewTerm(
        func=stance_root_orientation_error,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-400.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stance_upright = RewTerm(
        func=root_upright_violation,
        weight=-80.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "angle_threshold_deg": 4.0, "scale_deg": 12.0},
    )
    torso_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=TORSO_HEAD_LINKS),
            "min_clearance": 0.17,
            "active_only": False,
        },
    )
    torso_head_wall_proxy_clearance = RewTerm(
        func=wall_proxy_clearance_violation,
        weight=-16.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "min_clearance": 0.15,
            "scale": 0.12,
            "active_only": False,
        },
    )
    nonbrush_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=NONBRUSH_LINKS),
            "min_clearance": 0.12,
            "active_only": False,
        },
    )
    nonbrush_wall_contact = RewTerm(
        func=wall_contact_force_violation,
        weight=-80.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "threshold": 0.5,
            "near_wall_margin": 0.28,
        },
    )
    right_hand_wall_clearance = None
    left_arm_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_ARM_LINKS),
            "min_clearance": 0.12,
            "active_only": False,
        },
    )
    foot_reference = RewTerm(
        func=foot_stance_reference_violation,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINKS), "std": 0.035},
    )
    wall_facing_yaw = RewTerm(
        func=wall_facing_yaw_violation,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "yaw_threshold_deg": 15.0, "active_only": False},
    )


@configclass
class G1WallBrushNoWallStanceRewards(G1WallBrushStanceRewards):
    """Stand-first warmup for the virtual-wall task."""

    torso_wall_clearance = None
    torso_head_wall_proxy_clearance = None
    nonbrush_wall_clearance = None
    nonbrush_wall_contact = None
    left_arm_wall_clearance = None


@configclass
class TerminationsCfg(TerminationsCfgBase):
    torso_below_threshold = DoneTerm(func=root_below_threshold, params={"thres": 0.35})
    torso_angle_below_threshold = DoneTerm(func=root_angle_below_threshold, params={"thres": 0.35})
    self_collision_proxy = DoneTerm(
        func=self_collision_proxy_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pair_groups": SELF_COLLISION_HARD_RESET_PAIR_GROUPS,
            "tolerance": 0.03,
        },
    )
    nonbrush_wall_contact = DoneTerm(
        func=wall_contact_force_above_threshold,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "threshold": 0.5,
            "near_wall_margin": 0.28,
        },
    )
    torso_wall_clearance_below_threshold = DoneTerm(
        func=body_wall_clearance_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=TORSO_HEAD_LINKS),
            "min_clearance": 0.15,
            "active_only": False,
        },
    )
    torso_head_wall_proxy = DoneTerm(
        func=wall_proxy_clearance_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "min_clearance": 0.05,
            "active_only": False,
        },
    )
    nonbrush_wall_clearance_below_threshold = DoneTerm(
        func=body_wall_clearance_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=NONBRUSH_LINKS),
            "min_clearance": 0.08,
            "active_only": False,
        },
    )
    nonbrush_wall_penetration = None


@configclass
class WallBrushCurriculumEventCfg(EventCfg):
    reset_robot_joints = EventTerm(
        func=reset_joints_for_wall_brush_curriculum,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)},
    )
    reset_base = EventTerm(
        func=reset_root_state_for_wall_brush_stance,
        mode="reset",
        params={"offset_z": 0.15},
    )


@configclass
class WallBrushDreamControlEventCfg(WallBrushCurriculumEventCfg):
    reset_smoothness_buffers = EventTerm(
        func=reset_wall_brush_smoothness_buffers,
        mode="reset",
    )
    reset_base = EventTerm(
        func=reset_root_state_for_motion,
        mode="reset",
        params={"offset_z": 0.0},
    )


@configclass
class WallBrushCurriculumTerminationsCfg(TerminationsCfg):
    nonbrush_wall_penetration = None


@configclass
class WallBrushNoWallCollisionTerminationsCfg(TerminationsCfg):
    nonbrush_wall_contact = None
    torso_wall_clearance_below_threshold = None
    torso_head_wall_proxy = None
    nonbrush_wall_clearance_below_threshold = None


@configclass
class WallBrushDreamControlTerminationsCfg(WallBrushNoWallCollisionTerminationsCfg):
    """ButtonPress-style fall/tilt reset without self-collision reset."""

    torso_below_threshold = DoneTerm(func=root_below_threshold, params={"thres": 0.3})
    torso_angle_below_threshold = DoneTerm(func=root_angle_below_threshold, params={"thres": 0.5})
    self_collision_proxy = None


@configclass
class WallBrushStanceEventCfg(EventCfg):
    reset_robot_joints = EventTerm(
        func=reset_joints_for_wall_brush_stance,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)},
    )
    reset_base = EventTerm(
        func=reset_root_state_for_wall_brush_stance,
        mode="reset",
        params={"offset_z": 0.15},
    )


@configclass
class WallBrushStanceTerminationsCfg(TerminationsCfgBase):
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_below_threshold = DoneTerm(func=root_below_threshold, params={"thres": 0.35})
    torso_angle_below_threshold = DoneTerm(func=root_angle_below_threshold, params={"thres": 0.35})
    self_collision_proxy = DoneTerm(
        func=self_collision_proxy_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pair_groups": SELF_COLLISION_HARD_RESET_PAIR_GROUPS,
            "tolerance": 0.03,
        },
    )
    nonbrush_wall_contact = DoneTerm(
        func=wall_contact_force_above_threshold,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=ILLEGAL_WALL_CONTACT_LINKS, preserve_order=True),
            "threshold": 0.5,
            "near_wall_margin": 0.28,
        },
    )
    torso_wall_clearance_below_threshold = DoneTerm(
        func=body_wall_clearance_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=TORSO_HEAD_LINKS),
            "min_clearance": 0.15,
            "active_only": False,
        },
    )
    torso_head_wall_proxy = DoneTerm(
        func=wall_proxy_clearance_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "min_clearance": 0.05,
            "active_only": False,
        },
    )
    nonbrush_wall_clearance_below_threshold = DoneTerm(
        func=body_wall_clearance_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=NONBRUSH_LINKS),
            "min_clearance": 0.08,
            "active_only": False,
        },
    )
    nonbrush_wall_penetration = None


@configclass
class WallBrushStanceSoftGuardTerminationsCfg(WallBrushStanceTerminationsCfg):
    self_collision_proxy = None
    nonbrush_wall_contact = None
    torso_wall_clearance_below_threshold = None
    torso_head_wall_proxy = None
    nonbrush_wall_clearance_below_threshold = None


@configclass
class WallBrushUpperBodyEventCfg(EventCfgBase):
    """Official stationary-manipulation reset style: fixed root, full reference joints."""

    reset_success_buffers = EventTerm(
        func=reset_wall_brush_success_buffers,
        mode="reset",
    )
    reset_base = EventTerm(
        func=reset_root_state_for_motion,
        mode="reset",
        params={"offset_z": 0.01},
    )


@configclass
class WallBrushUpperBodyCurriculumEventCfg(WallBrushUpperBodyEventCfg):
    """Fixed-base reset with failure-prior weighted motion sampling."""

    reset_robot_joints = EventTerm(
        func=reset_joints_for_wall_brush_curriculum,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)},
    )


@configclass
class WallBrushUpperBodyTerminationsCfg(TerminationsCfgBase):
    """Fixed-base upper-body tasks should not reset from free-standing balance terms."""

    pass


@configclass
class WallBrushUpperBodyRightHandLegTerminationsCfg(TerminationsCfgBase):
    """Fixed-base upper-body task with right-hand/leg self-collision as an invalid rollout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    self_collision_proxy = DoneTerm(
        func=self_collision_proxy_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pair_groups": RIGHT_HAND_LEG_COLLISION_PAIR_GROUPS,
            "tolerance": 0.0,
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)
        target_ref_curr = ObsTerm(func=target_ref, params={"visualize_markers": VISUALIZE_MARKERS})
        target_ref_t_plus_0p1 = ObsTerm(func=target_ref, params={"time_offset": 0.1, "visualize_markers": VISUALIZE_MARKERS})
        target_ref_t_plus_0p2 = ObsTerm(func=target_ref, params={"time_offset": 0.2, "visualize_markers": VISUALIZE_MARKERS})
        target_ref_t_plus_0p3 = ObsTerm(func=target_ref, params={"time_offset": 0.3, "visualize_markers": VISUALIZE_MARKERS})
        target_ref_t_plus_0p4 = ObsTerm(func=target_ref, params={"time_offset": 0.4, "visualize_markers": VISUALIZE_MARKERS})
        wall_brush_target = ObsTerm(func=wall_brush_target_obs)
        current_time = ObsTerm(func=current_time_enc)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class WallBrushUpperBodyObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder_UB, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder_UB, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)
        target_ref_curr = ObsTerm(func=target_ref, params={"visualize_markers": VISUALIZE_MARKERS})
        target_ref_next = ObsTerm(func=target_ref, params={"time_offset": 0.1, "visualize_markers": VISUALIZE_MARKERS})
        wall_brush_target = ObsTerm(func=wall_brush_target_obs)
        current_time = ObsTerm(func=current_time_enc)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class WallBrushUpperBodyGuidedObservationsCfg(WallBrushUpperBodyObservationsCfg):
    @configclass
    class PolicyCfg(WallBrushUpperBodyObservationsCfg.PolicyCfg):
        brush_tip_error = ObsTerm(func=brush_tip_target_error_obs)

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg(ActionsCfgBase):
    pass


@configclass
class WallBrushUpperBodyActionsCfg(ActionsCfgBase):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JointNamesOrder_UB,
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class WallBrushRightArmUpperBodyActionsCfg(ActionsCfgBase):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "waist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class WallBrushSmallActionsCfg(ActionsCfgBase):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JointNamesOrder,
        preserve_order=True,
        scale=0.2,
        use_default_offset=True,
    )


@configclass
class WallBrushMotionResidualJointPositionActionCfg(mdp.JointPositionActionCfg):
    class_type: type = WallBrushMotionResidualJointPositionAction
    reference_mode: str = "current"
    reference_time_offset: float = 0.0


@configclass
class WallBrushAgileLowerBodyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = WallBrushAgileLowerBodyAction
    joint_names: list[str] = AGILE_LOWER_BODY_JOINTS
    policy_path: str = "/root/autodl-tmp/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher.pt"
    fixed_command: tuple[float, float, float, float] = (0.0, 0.05, 0.0, 0.70)
    policy_output_scale: dict[str, float] = AGILE_LOWER_BODY_POLICY_OUTPUT_SCALE


@configclass
class WallBrushRightArmCurrentResidualActionsCfg(ActionsCfgBase):
    joint_pos = WallBrushMotionResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "waist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        preserve_order=True,
        scale=0.08,
        use_default_offset=False,
        reference_mode="current",
    )


@configclass
class WallBrushAgileBaseActionsCfg(ActionsCfgBase):
    joint_pos = WallBrushMotionResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "waist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        preserve_order=True,
        scale=0.08,
        use_default_offset=False,
        reference_mode="current",
    )
    agile_lower_body = WallBrushAgileLowerBodyActionCfg(asset_name="robot")


@configclass
class WallBrushUpperBodyCurrentResidualActionsCfg(ActionsCfgBase):
    joint_pos = WallBrushMotionResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=JointNamesOrder_UB,
        preserve_order=True,
        scale=0.08,
        use_default_offset=False,
        reference_mode="current",
    )


@configclass
class WallBrushStanceResidualActionsCfg(ActionsCfgBase):
    joint_pos = WallBrushMotionResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=JointNamesOrder,
        preserve_order=True,
        scale=0.10,
        use_default_offset=False,
        reference_mode="stance",
    )


@configclass
class WallBrushStanceResidualWideActionsCfg(ActionsCfgBase):
    joint_pos = WallBrushMotionResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=JointNamesOrder,
        preserve_order=True,
        scale=0.50,
        use_default_offset=False,
        reference_mode="stance",
    )


@configclass
class WallBrushCurrentResidualActionsCfg(ActionsCfgBase):
    joint_pos = WallBrushMotionResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=JointNamesOrder,
        preserve_order=True,
        scale=0.10,
        use_default_offset=False,
        reference_mode="current",
    )


@configclass
class WallBrushCurrentResidualWideActionsCfg(ActionsCfgBase):
    joint_pos = WallBrushMotionResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=JointNamesOrder,
        preserve_order=True,
        scale=0.50,
        use_default_offset=False,
        reference_mode="current",
    )


@configclass
class MySceneCfg(MySceneCfgBase):
    wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.47, 0.0, 0.90), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.85, 0.75),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.88, 0.88, 0.82), metallic=0.0, opacity=0.45),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )


@configclass
class G1WallBrushEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.0)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol.npz"
        self.motion_lib_entry_point = "isaaclab_tasks.utils.motion_lib.wall_brush_motion_lib:WallBrushMotionLib"
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.47, 0.0, 0.90), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.85, 0.75),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.88, 0.88, 0.82), metallic=0.0, opacity=0.45),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    restitution_combine_mode="min",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        self.episode_length_s = 10.0


@configclass
class G1WallBrushSmokeEnvCfg(G1WallBrushEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=256, env_spacing=2.0)


@configclass
class G1WallBrushCurriculumEnvCfg(G1WallBrushEnvCfg):
    rewards: G1WallBrushCurriculumRewards = G1WallBrushCurriculumRewards()
    events: WallBrushCurriculumEventCfg = WallBrushCurriculumEventCfg()
    terminations: WallBrushCurriculumTerminationsCfg = WallBrushCurriculumTerminationsCfg()


@configclass
class G1WallBrushImitationEnvCfg(G1WallBrushEnvCfg):
    rewards: G1WallBrushImitationRewards = G1WallBrushImitationRewards()
    events: WallBrushCurriculumEventCfg = WallBrushCurriculumEventCfg()
    terminations: WallBrushCurriculumTerminationsCfg = WallBrushCurriculumTerminationsCfg()


@configclass
class G1WallBrushReachEnvCfg(G1WallBrushEnvCfg):
    rewards: G1WallBrushReachRewards = G1WallBrushReachRewards()
    events: WallBrushCurriculumEventCfg = WallBrushCurriculumEventCfg()
    terminations: WallBrushCurriculumTerminationsCfg = WallBrushCurriculumTerminationsCfg()


@configclass
class G1WallBrushActionPriorEnvCfg(G1WallBrushEnvCfg):
    rewards: G1WallBrushActionPriorRewards = G1WallBrushActionPriorRewards()
    events: WallBrushCurriculumEventCfg = WallBrushCurriculumEventCfg()
    terminations: WallBrushCurriculumTerminationsCfg = WallBrushCurriculumTerminationsCfg()


@configclass
class G1WallBrushNoWallCollisionEnvCfg(G1WallBrushEnvCfg):
    rewards: G1WallBrushNoWallCollisionRewards = G1WallBrushNoWallCollisionRewards()
    events: WallBrushCurriculumEventCfg = WallBrushCurriculumEventCfg()
    terminations: WallBrushNoWallCollisionTerminationsCfg = WallBrushNoWallCollisionTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.47, 0.0, 0.90), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.85, 0.75),
                collision_props=None,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.88, 0.88, 0.82), metallic=0.0, opacity=0.35),
            ),
        )


@configclass
class G1WallBrushNoWallCollisionDreamControlEnvCfg(G1WallBrushNoWallCollisionEnvCfg):
    """Full-body DreamControl-style wall brush with virtual wall target and staged250 priors."""

    actions: ActionsCfg = ActionsCfg()
    events: WallBrushDreamControlEventCfg = WallBrushDreamControlEventCfg()
    terminations: WallBrushDreamControlTerminationsCfg = WallBrushDreamControlTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.ref_motions_path = (
            "../TrajGen/sample/Wall_Brush_27/"
            "wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"
        )
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005


@configclass
class G1WallBrushNoWallCollisionDreamControlWarmstartEnvCfg(G1WallBrushNoWallCollisionDreamControlEnvCfg):
    """Full-body root-unlocked warm-start stage for staged250 wall brushing."""

    actions: ActionsCfg = ActionsCfg()
    rewards: G1WallBrushDreamControlWarmstartRewards = G1WallBrushDreamControlWarmstartRewards()
    terminations: WallBrushDreamControlTerminationsCfg = WallBrushDreamControlTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005


@configclass
class G1WallBrushNoWallCollisionDreamControlStandStillRowContactEnvCfg(G1WallBrushNoWallCollisionDreamControlEnvCfg):
    """Full-body Stand_Still continuation stage with warmstart stability and gentle row/contact rewards."""

    actions: ActionsCfg = ActionsCfg()
    rewards: G1WallBrushDreamControlStandStillRowContactRewards = G1WallBrushDreamControlStandStillRowContactRewards()
    terminations: WallBrushDreamControlTerminationsCfg = WallBrushDreamControlTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005


@configclass
class G1WallBrushNoWallCollisionDreamControlBalanceWarmstartEnvCfg(G1WallBrushNoWallCollisionDreamControlEnvCfg):
    """Full-body root-unlocked balance curriculum before 10s wall-brush training."""

    actions: ActionsCfg = ActionsCfg()
    rewards: G1WallBrushDreamControlBalanceWarmstartRewards = G1WallBrushDreamControlBalanceWarmstartRewards()
    terminations: WallBrushDreamControlTerminationsCfg = WallBrushDreamControlTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 0.005


@configclass
class G1WallBrushNoWallCollisionDreamControlButtonPressAlignedEnvCfg(G1WallBrushNoWallCollisionDreamControlEnvCfg):
    """Full-body wall-brush task with ButtonPress-scale rewards and the staged250 prior."""

    actions: ActionsCfg = ActionsCfg()
    rewards: G1WallBrushDreamControlButtonPressAlignedRewards = G1WallBrushDreamControlButtonPressAlignedRewards()
    terminations: WallBrushDreamControlTerminationsCfg = WallBrushDreamControlTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005


@configclass
class G1WallBrushNoWallCollisionDreamControlButtonPressAlignedAntiJitterEnvCfg(G1WallBrushNoWallCollisionDreamControlButtonPressAlignedEnvCfg):
    """ButtonPressAligned full-body task with anti-jitter and collision-free shaping."""

    rewards: G1WallBrushDreamControlButtonPressAlignedAntiJitterRewards = (
        G1WallBrushDreamControlButtonPressAlignedAntiJitterRewards()
    )


@configclass
class G1WallBrushNoWallCollisionDreamControlButtonPressBalanceEnvCfg(G1WallBrushNoWallCollisionDreamControlEnvCfg):
    """Short-horizon full-body standing curriculum using ButtonPress-scale PPO/rewards."""

    actions: ActionsCfg = ActionsCfg()
    rewards: G1WallBrushDreamControlButtonPressBalanceRewards = G1WallBrushDreamControlButtonPressBalanceRewards()
    terminations: WallBrushDreamControlTerminationsCfg = WallBrushDreamControlTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 0.005


@configclass
class G1WallBrushNoWallCollisionDreamControlAgileBaseEnvCfg(G1WallBrushNoWallCollisionDreamControlEnvCfg):
    """Root-unlocked wall-brush task with frozen AGILE legs and trainable right-arm residual control."""

    actions: WallBrushAgileBaseActionsCfg = WallBrushAgileBaseActionsCfg()
    rewards: G1WallBrushDreamControlAgileBaseRewards = G1WallBrushDreamControlAgileBaseRewards()
    terminations: WallBrushDreamControlTerminationsCfg = WallBrushDreamControlTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005


@configclass
class G1WallBrushStanceEnvCfg(G1WallBrushEnvCfg):
    rewards: G1WallBrushStanceRewards = G1WallBrushStanceRewards()
    events: WallBrushStanceEventCfg = WallBrushStanceEventCfg()
    terminations: WallBrushStanceTerminationsCfg = WallBrushStanceTerminationsCfg()


@configclass
class G1WallBrushNoWallStanceEnvCfg(G1WallBrushStanceEnvCfg):
    rewards: G1WallBrushNoWallStanceRewards = G1WallBrushNoWallStanceRewards()
    terminations: WallBrushStanceSoftGuardTerminationsCfg = WallBrushStanceSoftGuardTerminationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.47, 0.0, 0.90), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.85, 0.75),
                collision_props=None,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.88, 0.88, 0.82), metallic=0.0, opacity=0.35),
            ),
        )


@configclass
class G1WallBrushNoWallStanceSmallActionEnvCfg(G1WallBrushNoWallStanceEnvCfg):
    actions: WallBrushSmallActionsCfg = WallBrushSmallActionsCfg()


@configclass
class G1WallBrushNoWallStanceResidualActionEnvCfg(G1WallBrushNoWallStanceEnvCfg):
    actions: WallBrushStanceResidualActionsCfg = WallBrushStanceResidualActionsCfg()


@configclass
class G1WallBrushNoWallStanceResidualWideActionEnvCfg(G1WallBrushNoWallStanceEnvCfg):
    actions: WallBrushStanceResidualWideActionsCfg = WallBrushStanceResidualWideActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_standfix_ik.npz"


@configclass
class G1WallBrushNoWallStanceStandPrepIKResidualWideActionEnvCfg(G1WallBrushNoWallStanceEnvCfg):
    """Stand-only warmup with DreamControl-style default-pose motion prepending."""

    actions: WallBrushStanceResidualWideActionsCfg = WallBrushStanceResidualWideActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_standprep_ik.npz"


@configclass
class G1WallBrushNoWallCollisionResidualActionEnvCfg(G1WallBrushNoWallCollisionEnvCfg):
    actions: WallBrushCurrentResidualActionsCfg = WallBrushCurrentResidualActionsCfg()


@configclass
class G1WallBrushNoWallCollisionOfficialResetEnvCfg(G1WallBrushNoWallCollisionEnvCfg):
    """DreamControl-style no-wall task that keeps the official motion root reset."""

    events: EventCfg = EventCfg()


@configclass
class G1WallBrushNoWallCollisionOfficialResetResidualActionEnvCfg(G1WallBrushNoWallCollisionOfficialResetEnvCfg):
    actions: WallBrushCurrentResidualActionsCfg = WallBrushCurrentResidualActionsCfg()


@configclass
class G1WallBrushNoWallCollisionStandFixResidualActionEnvCfg(G1WallBrushNoWallCollisionEnvCfg):
    """No-wall task using the stand-first repaired prior bundle."""

    actions: WallBrushCurrentResidualActionsCfg = WallBrushCurrentResidualActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_standfix.npz"


@configclass
class G1WallBrushNoWallCollisionStandFixIKResidualActionEnvCfg(G1WallBrushNoWallCollisionStandFixResidualActionEnvCfg):
    """No-wall task using stand-first prior plus right-arm IK refinement."""

    def __post_init__(self):
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_standfix_ik.npz"


@configclass
class G1WallBrushNoWallCollisionStandFixIKResidualWideActionEnvCfg(G1WallBrushNoWallCollisionStandFixIKResidualActionEnvCfg):
    """StandFixIK task with G1-style residual authority for active balance."""

    actions: WallBrushCurrentResidualWideActionsCfg = WallBrushCurrentResidualWideActionsCfg()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKResidualWideActionEnvCfg(G1WallBrushNoWallCollisionStandFixIKResidualWideActionEnvCfg):
    """No-wall brush task using a default-pose prepended StandFixIK prior."""

    def __post_init__(self):
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_standprep_ik.npz"


@configclass
class G1WallBrushNoWallCollisionStandPrepIKUpperBodyEnvCfg(G1WallBrushNoWallCollisionEnvCfg):
    """Official DreamControl-style fixed-base upper-body wall-brush task."""

    rewards: G1WallBrushUpperBodyRewards = G1WallBrushUpperBodyRewards()
    events: WallBrushUpperBodyEventCfg = WallBrushUpperBodyEventCfg()
    observations: WallBrushUpperBodyObservationsCfg = WallBrushUpperBodyObservationsCfg()
    actions: WallBrushUpperBodyActionsCfg = WallBrushUpperBodyActionsCfg()
    terminations: WallBrushUpperBodyTerminationsCfg = WallBrushUpperBodyTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_standprep_ik.npz"
        self.scene.robot = G1_MINIMAL_CFG_FIXED_BASE.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005


@configclass
class G1WallBrushNoWallCollisionStandPrepIKRightArmUpperBodyEnvCfg(G1WallBrushNoWallCollisionStandPrepIKUpperBodyEnvCfg):
    """Fixed-base wall-brush task with only waist and right-arm actions."""

    actions: WallBrushRightArmUpperBodyActionsCfg = WallBrushRightArmUpperBodyActionsCfg()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKRightArmOnlyRewardEnvCfg(G1WallBrushNoWallCollisionStandPrepIKRightArmUpperBodyEnvCfg):
    """Right-arm action task with reference rewards masked to waist, torso, and right arm."""

    rewards: G1WallBrushRightArmUpperBodyRewards = G1WallBrushRightArmUpperBodyRewards()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKRightArmAnchorRewardEnvCfg(G1WallBrushNoWallCollisionStandPrepIKRightArmOnlyRewardEnvCfg):
    """Right-arm task with DreamControl-style approach reward toward ordered brush anchors."""

    rewards: G1WallBrushRightArmAnchorRewards = G1WallBrushRightArmAnchorRewards()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKRightArmResidualAnchorEnvCfg(G1WallBrushNoWallCollisionStandPrepIKRightArmAnchorRewardEnvCfg):
    """Right-arm fixed-base task with residual actions centered on the motion prior."""

    actions: WallBrushRightArmCurrentResidualActionsCfg = WallBrushRightArmCurrentResidualActionsCfg()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorEnvCfg(G1WallBrushNoWallCollisionStandPrepIKRightArmAnchorRewardEnvCfg):
    """Upper-body residual task centered on the motion prior, scored by right-hand brush success."""

    actions: WallBrushUpperBodyCurrentResidualActionsCfg = WallBrushUpperBodyCurrentResidualActionsCfg()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorRightHandLegResetEnvCfg(
    G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorEnvCfg
):
    """Upper-body residual task that invalidates right-hand/leg self-collision."""

    terminations: WallBrushUpperBodyRightHandLegTerminationsCfg = WallBrushUpperBodyRightHandLegTerminationsCfg()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorRegularizedEnvCfg(
    G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorEnvCfg
):
    """Upper-body residual-anchor task with action-L2 regularization against residual drift."""

    rewards: G1WallBrushResidualAnchorRegularizedRewards = G1WallBrushResidualAnchorRegularizedRewards()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorRegularizedCurriculumEnvCfg(
    G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorRegularizedEnvCfg
):
    """Regularized upper-body residual task with weighted sampling of harder wall-brush priors."""

    events: WallBrushUpperBodyCurriculumEventCfg = WallBrushUpperBodyCurriculumEventCfg()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorRegularizedGuidedEnvCfg(
    G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorRegularizedCurriculumEnvCfg
):
    """Regularized residual task with explicit brush-tip target-error observation."""

    observations: WallBrushUpperBodyGuidedObservationsCfg = WallBrushUpperBodyGuidedObservationsCfg()


@configclass
class G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorTargetedLowRowEnvCfg(
    G1WallBrushNoWallCollisionStandPrepIKUpperBodyResidualAnchorRegularizedCurriculumEnvCfg
):
    """Target low/near brush rows while explicitly discouraging right-hand/leg self collision."""

    rewards: G1WallBrushTargetedLowRowRewards = G1WallBrushTargetedLowRowRewards()


@configclass
class G1WallBrushStanceSoftGuardEnvCfg(G1WallBrushStanceEnvCfg):
    terminations: WallBrushStanceSoftGuardTerminationsCfg = WallBrushStanceSoftGuardTerminationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.0)

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
