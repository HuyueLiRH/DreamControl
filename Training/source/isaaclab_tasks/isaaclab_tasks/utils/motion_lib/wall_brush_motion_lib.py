from __future__ import annotations

import os.path as osp

import numpy as np
import torch
import pytorch_kinematics as pk2

from isaac_utils.rotations import quaternion_to_matrix, slerp
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder


RIGHT_WRIST_YAW_KEYPOINT_INDEX = 37
PLAYBACK_SPEED = 1.0


def _to_torch(array: np.ndarray, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


class WallBrushMotionLib:
    """Loads the 27 accepted postprocessed wall-brush priors exported as one NPZ bundle."""

    def __init__(self, num_envs, device, motion_file, sim_fps=50):
        self.num_envs = num_envs
        self._device = device
        self._sim_fps = sim_fps
        self.motion_file = motion_file
        self.joint_names = JointNamesOrder

    def _resolve_bundle_path(self) -> str:
        if osp.isdir(self.motion_file):
            return osp.join(self.motion_file, "wall_brush_27_prior_references_dreamcontrol.npz")
        return self.motion_file

    def _get_keypts(self, joint_angles: torch.Tensor, pk2_robot) -> torch.Tensor:
        q_dict = {name: joint_angles[:, i] for i, name in enumerate(self.joint_names)}
        tf_dict = pk2_robot.forward_kinematics(q_dict)
        keypts = torch.zeros((len(joint_angles), len(tf_dict), 3), dtype=joint_angles.dtype)
        for idx, name in enumerate(tf_dict.keys()):
            keypts[:, idx, :] = tf_dict[name].get_matrix()[:, :3, -1]
        return keypts

    def _transform_keypts(self, keypts: torch.Tensor, quat: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        rot_matrix = quaternion_to_matrix(quat)
        return torch.bmm(keypts, rot_matrix.transpose(2, 1)) + translation.unsqueeze(1)

    def load_motions(self):
        bundle_path = self._resolve_bundle_path()
        data = np.load(bundle_path, allow_pickle=False)
        required = [
            "dreamcontrol_root_pos",
            "dreamcontrol_root_rot",
            "dreamcontrol_dof_pos",
            "dreamcontrol_true_wall_points",
            "constraint_frames",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{bundle_path} is missing DreamControl fields: {', '.join(missing)}")

        root_pos = np.asarray(data["dreamcontrol_root_pos"], dtype=np.float32)
        root_rot = np.asarray(data["dreamcontrol_root_rot"], dtype=np.float32)
        dof_pos = np.asarray(data["dreamcontrol_dof_pos"], dtype=np.float32)
        wall_points = np.asarray(data["dreamcontrol_true_wall_points"], dtype=np.float32)
        constraint_frames = np.asarray(data["constraint_frames"], dtype=np.int64)
        fps = float(np.asarray(data["fps"]).reshape(())) if "fps" in data.files else 30.0
        brush_offset = (
            np.asarray(data["dreamcontrol_brush_offset"], dtype=np.float32)
            if "dreamcontrol_brush_offset" in data.files
            else np.asarray([0.17, 0.0, 0.0], dtype=np.float32)
        )

        if dof_pos.shape[-1] != len(JointNamesOrder):
            raise ValueError(f"Expected 27 DreamControl DoF columns, got {dof_pos.shape}")

        motions, frames = dof_pos.shape[:2]
        urdf_path = "HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
        pk2_robot = pk2.build_chain_from_urdf(open(urdf_path).read())
        flat_dof = torch.as_tensor(dof_pos.reshape(motions * frames, len(JointNamesOrder)), dtype=torch.float32)
        flat_keypts = self._get_keypts(flat_dof, pk2_robot)
        local_keypts = flat_keypts.reshape(motions, frames, flat_keypts.shape[1], 3)
        global_keypts = self._transform_keypts(
            local_keypts.reshape(motions * frames, local_keypts.shape[2], 3),
            torch.as_tensor(root_rot.reshape(motions * frames, 4), dtype=torch.float32),
            torch.as_tensor(root_pos.reshape(motions * frames, 3), dtype=torch.float32),
        ).reshape(motions, frames, local_keypts.shape[2], 3)

        if "dreamcontrol_brush_tip_ref" in data.files:
            brush_tip_ref = np.asarray(data["dreamcontrol_brush_tip_ref"], dtype=np.float32)
            if brush_tip_ref.shape != root_pos.shape:
                raise ValueError(
                    f"dreamcontrol_brush_tip_ref shape {brush_tip_ref.shape} does not match root_pos shape {root_pos.shape}"
                )
            # The exported brush target is the virtual task target on the wall plane. Recomputing it from
            # retargeted joints can move the target behind the wall and conflict with wall-contact rewards.
            wrist_tip_ref = torch.as_tensor(brush_tip_ref, dtype=torch.float32)
        else:
            wrist_tip_ref = global_keypts[:, :, RIGHT_WRIST_YAW_KEYPOINT_INDEX, :] + torch.as_tensor(brush_offset)[
                None, None, :
            ]
        stroke_start = constraint_frames[:, 0]
        stroke_end = constraint_frames[:, -1]
        midpoint = wall_points[:, len(wall_points[0]) // 2, :]

        self.transl = _to_torch(root_pos, self._device)
        self.quats = _to_torch(root_rot, self._device)
        self.dof_pos = _to_torch(dof_pos, self._device)
        self.local_keypts = local_keypts.to(self._device).float()
        self.global_keypts = global_keypts.to(self._device).float()
        self.wall_points = _to_torch(wall_points, self._device)
        self.brush_tip_ref = wrist_tip_ref.to(self._device).float()
        self.brush_offset = _to_torch(brush_offset, self._device)
        self.grab_pos = _to_torch(midpoint, self._device)
        self.offsets = torch.zeros((motions, 3), dtype=torch.float32, device=self._device)
        self.stroke_start_idxs = torch.as_tensor(stroke_start, dtype=torch.long, device=self._device)
        self.stroke_end_idxs = torch.as_tensor(stroke_end, dtype=torch.long, device=self._device)
        self.switch_idxs = self.stroke_start_idxs.float()
        self._motion_dt = torch.full((motions,), 1.0 / (fps * PLAYBACK_SPEED), dtype=torch.float32, device=self._device)
        self._motion_num_frames = torch.full((motions,), frames, dtype=torch.int32, device=self._device)
        self._motion_lengths = self._motion_num_frames.float() * self._motion_dt
        self._num_motions = motions
        print(f"Loaded {self._num_motions} wall-brush prior motions from {bundle_path}")

    def num_motions(self):
        return self._num_motions

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        return self._motion_lengths[motion_ids]

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            motion_len = motion_len - truncate_time
        return phase * motion_len

    def _calc_frame_blend(self, time, motion_len, num_frames, dt):
        time = torch.clamp(time.clone(), min=0.0)
        phase = torch.clip(time / motion_len, 0.0, 1.0)
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)
        return frame_idx0, frame_idx1, blend

    def get_motion_state(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        root_pos0 = self.transl[motion_ids, frame_idx0, :]
        root_pos1 = self.transl[motion_ids, frame_idx1, :]
        root_rot0 = self.quats[motion_ids, frame_idx0, :]
        root_rot1 = self.quats[motion_ids, frame_idx1, :]
        dof_pos0 = self.dof_pos[motion_ids, frame_idx0, :]
        dof_pos1 = self.dof_pos[motion_ids, frame_idx1, :]
        local_keypts0 = self.local_keypts[motion_ids, frame_idx0, :, :]
        local_keypts1 = self.local_keypts[motion_ids, frame_idx1, :, :]
        global_keypts0 = self.global_keypts[motion_ids, frame_idx0, :, :]
        global_keypts1 = self.global_keypts[motion_ids, frame_idx1, :, :]
        brush_tip0 = self.brush_tip_ref[motion_ids, frame_idx0, :]
        brush_tip1 = self.brush_tip_ref[motion_ids, frame_idx1, :]

        blend_pos = blend.unsqueeze(-1)
        blend_keypts = blend_pos.unsqueeze(-1)
        root_pos = (1.0 - blend_pos) * root_pos0 + blend_pos * root_pos1
        dof_pos = (1.0 - blend_pos) * dof_pos0 + blend_pos * dof_pos1
        local_keypts = (1.0 - blend_keypts) * local_keypts0 + blend_keypts * local_keypts1
        global_keypts = (1.0 - blend_keypts) * global_keypts0 + blend_keypts * global_keypts1
        brush_tip = (1.0 - blend_pos) * brush_tip0 + blend_pos * brush_tip1
        root_rot = slerp(root_rot0, root_rot1, blend_pos)

        stroke_active = (frame_idx0 >= self.stroke_start_idxs[motion_ids]) & (frame_idx0 <= self.stroke_end_idxs[motion_ids])
        wall_points = self.wall_points[motion_ids]

        return {
            "root_pos": root_pos.clone(),
            "root_rot": root_rot.clone(),
            "dof_pos": dof_pos.clone(),
            "local_keypts": local_keypts.clone(),
            "global_keypts": global_keypts.clone(),
            "grab_pos": self.grab_pos[motion_ids].clone(),
            "offsets": self.offsets[motion_ids].clone(),
            "is_closed": stroke_active,
            "stroke_active": stroke_active,
            "brush_tip_pos": brush_tip.clone(),
            "brush_offset": self.brush_offset.unsqueeze(0).repeat(len(motion_ids), 1),
            "wall_points": wall_points.clone(),
            "wall_start": wall_points[:, 0, :].clone(),
            "wall_mid": wall_points[:, wall_points.shape[1] // 2, :].clone(),
            "wall_end": wall_points[:, -1, :].clone(),
        }
