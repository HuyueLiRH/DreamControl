import pyroki as pk
import jax.numpy as jnp
import numpy as onp
from typing import TypedDict
import jaxlie
import jax
import jax.numpy as jnp
import jaxls
from pyroki.collision import colldist_from_sdf, collide
from typing import Tuple, TypedDict
import jax_dataclasses as jdc
import torch
from isaac_utils.rotations import quaternion_to_matrix

class RetargetingWeights(TypedDict):
    local_alignment: float
    """Local alignment weight, by matching the relative joint/keypoint positions and angles."""
    global_alignment: float
    """Global alignment weight, by matching the keypoint positions to the robot."""
    floor_contact: float
    """Floor contact weight, to place the robot's foot on the floor."""
    root_smoothness: float
    """Root smoothness weight, to penalize the robot's root from jittering too much."""
    foot_skating: float
    """Foot skating weight, to penalize the robot's foot from moving when it is in contact with the floor."""
    world_collision: float
    """World collision weight, to penalize the robot from colliding with the world."""

def get_keypts(joint_angles, joint_names, pk2_robot):
    q_dict = {name: joint_angles[:, i] for i, name in enumerate( joint_names ) }
    tf_dict = pk2_robot.forward_kinematics( q_dict )
    keypts = torch.zeros((len(joint_angles), len(tf_dict), 3))
    cntr = 0 
    
    for name in tf_dict.keys():
        tf_val = tf_dict[name].get_matrix()  
        t = tf_val[:, :3, -1 ]
        keypts[:, cntr , :] = t 
        cntr += 1 
    return keypts


def transform_keypts(keypts, quat, translation):
    """
    Transform keypoints using a quaternion and translation.
    Args:
        keypts: Tensor of shape (N, K, 3) where N is the number of samples, K is the number of keypoints.
        quat: Tensor of shape (N, 4) representing the quaternion.
        translation: Tensor of shape (N, 3) representing the translation.
    Returns:
        Transformed keypoints of shape (N, K, 3).
    """
    # Convert quaternion to rotation matrix (final shape will be (N, 3, 3))
    rot_matrix = quaternion_to_matrix(quat)
    # Ensure keypts is of shape (N, K, 3)
    if keypts.dim() == 2:
        keypts = keypts.unsqueeze(1)
    elif keypts.dim() != 3 or keypts.shape[-1] != 3:
        raise ValueError("keypts must be of shape (N, K, 3) or (N, 3)")
    
    # Apply rotation and translation
    transformed_keypts = torch.bmm(keypts, rot_matrix.transpose(2, 1)) + translation.unsqueeze(1)
    return transformed_keypts

def foot_detect(positions, thres):
    fid_r, fid_l = [8, 11], [7, 10]

    velfactor, heightfactor = onp.array([thres, thres]), onp.array([3.0, 2.0])

    feet_l_x = (positions[:, 1:, fid_l, 0] - positions[:, :-1, fid_l, 0]) ** 2
    feet_l_y = (positions[:, 1:, fid_l, 1] - positions[:, :-1, fid_l, 1]) ** 2
    feet_l_z = (positions[:, 1:, fid_l, 2] - positions[:, :-1, fid_l, 2]) ** 2
    feet_l = onp.sum(((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(onp.float32),axis=-1)

    feet_r_x = (positions[:, 1:, fid_r, 0] - positions[:, :-1, fid_r, 0]) ** 2
    feet_r_y = (positions[:, 1:, fid_r, 1] - positions[:, :-1, fid_r, 1]) ** 2
    feet_r_z = (positions[:, 1:, fid_r, 2] - positions[:, :-1, fid_r, 2]) ** 2
    feet_r = onp.sum(((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(onp.float32),axis=-1)
    return feet_l, feet_r

@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    target_keypoints: jnp.ndarray,
    is_left_foot_contact: jnp.ndarray,
    is_right_foot_contact: jnp.ndarray,
    left_foot_keypoints: jnp.ndarray,
    right_foot_keypoints: jnp.ndarray,
    smpl_joint_retarget_indices: jnp.ndarray,
    g1_joint_retarget_indices: jnp.ndarray,
    smpl_mask: jnp.ndarray,
    heightmap: pk.collision.Heightmap,
    weights: RetargetingWeights,
    pick_point: jnp.ndarray,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the retargeting problem."""

    n_retarget = len(smpl_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    # Robot properties.
    # - Joints that should move less for natural humanoid motion.

    joints_to_move_less = jnp.array(
        [
            robot.joints.actuated_names.index(name)
            for name in ["left_hip_yaw_joint", "right_hip_yaw_joint"]
        ]
    )



    # - Foot indices.
    left_foot_idx = robot.links.names.index("left_ankle_roll_link")
    right_foot_idx = robot.links.names.index("right_ankle_roll_link")

    # Variables.
    class SmplJointsScaleVarG1(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((n_retarget, n_retarget))
    ): ...

    class OffsetVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros((3,))): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_smpl_joints_scale = SmplJointsScaleVarG1(jnp.zeros(timesteps))
    var_offset = OffsetVar(jnp.zeros(timesteps))

    # Costs.
    costs: list[jaxls.Cost] = []

    @jaxls.Cost.create_factory
    def avoid_table_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_smpl_joints_scale: SmplJointsScaleVarG1,
        keypoints: jnp.ndarray,
        table_y: float = 0.8,
        table_z: float = 1.0,
    ) -> jax.Array:
        """Retargeting factor, with a focus on:
        - matching the relative joint/keypoint positions (vectors).
        - and matching the relative angles between the vectors.
        """
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link

        cost = weights["avoid_table"]*(T_world_link.translation()[:, 1] < table_y) * (T_world_link.translation()[:, 2] > table_z)
        smpl_pos = keypoints[jnp.array(smpl_joint_retarget_indices)]
        robot_pos = T_world_link.translation()[jnp.array(g1_joint_retarget_indices)]

        # NxN grid of relative positions.
        delta_smpl = smpl_pos[:, None] - smpl_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        # Vector regularization.
        position_scale = var_values[var_smpl_joints_scale][..., None]
        residual_position_delta = (
            (delta_smpl - delta_robot * position_scale)
            * (1 - jnp.eye(delta_smpl.shape[0])[..., None])
            * smpl_mask[..., None]
        )

        # Vector angle regularization.
        delta_smpl_normalized = delta_smpl / jnp.linalg.norm(
            delta_smpl + 1e-6, axis=-1, keepdims=True
        )
        delta_robot_normalized = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True
        )
        residual_angle_delta = 1 - (delta_smpl_normalized * delta_robot_normalized).sum(
            axis=-1
        )
        residual_angle_delta = (
            residual_angle_delta
            * (1 - jnp.eye(residual_angle_delta.shape[0]))
            * smpl_mask
        )

        residual = (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()]
            )
            * weights["local_alignment"]
        )
        return residual


    @jaxls.Cost.create_factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_smpl_joints_scale: SmplJointsScaleVarG1,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Retargeting factor, with a focus on:
        - matching the relative joint/keypoint positions (vectors).
        - and matching the relative angles between the vectors.
        """
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link

        smpl_pos = keypoints[jnp.array(smpl_joint_retarget_indices)]
        robot_pos = T_world_link.translation()[jnp.array(g1_joint_retarget_indices)]

        # NxN grid of relative positions.
        delta_smpl = smpl_pos[:, None] - smpl_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        # Vector regularization.
        position_scale = var_values[var_smpl_joints_scale][..., None]
        residual_position_delta = (
            (delta_smpl - delta_robot * position_scale)
            * (1 - jnp.eye(delta_smpl.shape[0])[..., None])
            * smpl_mask[..., None]
        )

        # Vector angle regularization.
        delta_smpl_normalized = delta_smpl / jnp.linalg.norm(
            delta_smpl + 1e-6, axis=-1, keepdims=True
        )
        delta_robot_normalized = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True
        )
        residual_angle_delta = 1 - (delta_smpl_normalized * delta_robot_normalized).sum(
            axis=-1
        )
        residual_angle_delta = (
            residual_angle_delta
            * (1 - jnp.eye(residual_angle_delta.shape[0]))
            * smpl_mask
        )

        residual = (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()]
            )
            * weights["local_alignment"]
        )
        return residual

    @jaxls.Cost.create_factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_smpl_joints_scale: SmplJointsScaleVarG1,
    ) -> jax.Array:
        """Regularize the scale of the retargeted joints."""
        # Close to 1.
        res_0 = (var_values[var_smpl_joints_scale] - 1.0).flatten() * 1.0
        # Symmetric.
        res_1 = (
            var_values[var_smpl_joints_scale] - var_values[var_smpl_joints_scale].T
        ).flatten() * 100.0
        # Non-negative.
        res_2 = jnp.clip(-var_values[var_smpl_joints_scale], min=0).flatten() * 100.0
        return jnp.concatenate([res_0, res_1, res_2])

    @jaxls.Cost.create_factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Soft cost to align the human keypoints to the robot, in the world frame."""
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[g1_joint_retarget_indices]
        keypoint_pos = keypoints[smpl_joint_retarget_indices]
        return (link_pos - keypoint_pos).flatten() * weights["global_alignment"]

    @jaxls.Cost.create_factory
    def floor_contact_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_offset: OffsetVar,
        is_left_foot_contact: jnp.ndarray,
        is_right_foot_contact: jnp.ndarray,
        left_foot_keypoints: jnp.ndarray,
        right_foot_keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Cost to place the robot on the floor:
        - match foot keypoint positions, and
        - penalize the foot from tilting too much.
        """
        T_world_root = var_values[var_Ts_world_root]
        T_root_link = jaxlie.SE3(
            robot.forward_kinematics(cfg=var_values[var_robot_cfg])
        )

        offset = var_values[var_offset]
        left_foot_pos = (T_world_root @ T_root_link).translation()[
            left_foot_idx
        ] + offset
        right_foot_pos = (T_world_root @ T_root_link).translation()[
            right_foot_idx
        ] + offset
        left_foot_contact_cost = (
            is_left_foot_contact * (left_foot_pos - left_foot_keypoints) ** 2
        )
        right_foot_contact_cost = (
            is_right_foot_contact * (right_foot_pos - right_foot_keypoints) ** 2
        )

        # Also penalize the foot from tilting too much -- keep z axis up!
        left_foot_ori = (
            (T_world_root @ T_root_link).rotation().as_matrix()[left_foot_idx]
        )
        right_foot_ori = (
            (T_world_root @ T_root_link).rotation().as_matrix()[right_foot_idx]
        )
        left_foot_contact_residual_rot = jnp.where(
            is_left_foot_contact,
            left_foot_ori[2, 2] - 1,
            0.0,
        )
        right_foot_contact_residual_rot = jnp.where(
            is_right_foot_contact,
            right_foot_ori[2, 2] - 1,
            0.0,
        )
        
        return (
            jnp.concatenate(
                [
                    left_foot_contact_cost.flatten(),
                    right_foot_contact_cost.flatten(),
                    left_foot_contact_residual_rot.flatten(),
                    right_foot_contact_residual_rot.flatten(),
                ]
            )
            * weights["floor_contact"]
        )

    @jaxls.Cost.create_factory
    def root_smoothness(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        """Smoothness cost for the robot root pose."""
        return (
            var_values[var_Ts_world_root].inverse() @ var_values[var_Ts_world_root_prev]
        ).log().flatten() * weights["root_smoothness"]

    @jaxls.Cost.create_factory
    def skating_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_offset: OffsetVar,
        var_Ts_world_root_prev: jaxls.SE3Var,
        var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
        var_offset_prev: OffsetVar,
        is_left_foot_contact: jnp.ndarray,
        is_right_foot_contact: jnp.ndarray,
    ) -> jax.Array:
        """Cost to penalize the robot for skating."""
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        offset = var_values[var_offset]
        T_link = T_world_root @ T_root_link
        left_foot_pos = T_link.translation()[left_foot_idx] + offset
        right_foot_pos = T_link.translation()[right_foot_idx] + offset

        T_world_root_prev = var_values[var_Ts_world_root_prev]
        robot_cfg_prev = var_values[var_robot_cfg_prev]
        T_root_link_prev = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_prev))
        offset_prev = var_values[var_offset_prev]
        T_link_prev = T_world_root_prev @ T_root_link_prev
        left_foot_pos_prev = T_link_prev.translation()[left_foot_idx] + offset_prev
        right_foot_pos_prev = T_link_prev.translation()[right_foot_idx] + offset_prev

        skating_cost_left = is_left_foot_contact * (left_foot_pos - left_foot_pos_prev)
        skating_cost_right = is_right_foot_contact * (
            right_foot_pos - right_foot_pos_prev
        )

        return (
            jnp.stack([skating_cost_left, skating_cost_right]) * weights["foot_skating"]
        )

    @jaxls.Cost.create_factory
    def world_collision_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_offset: OffsetVar,
    ) -> jax.Array:
        """
        World collision; we intentionally use a low weight --
        high enough to lift the robot up from the ground, but
        low enough to not interfere with the retargeting.
        """
        Ts_world_root = var_values[var_Ts_world_root]
        T_offset = jaxlie.SE3.from_translation(var_values[var_offset])
        transform = T_offset @ Ts_world_root

        robot_cfg = var_values[var_robot_cfg]
        coll = robot_coll.at_config(robot, robot_cfg)
        coll = coll.transform(transform)

        dist = collide(coll, heightmap)
        act = colldist_from_sdf(dist, activation_dist=0.005)
        return act.flatten() * weights["world_collision"]

    costs = [
        # Costs that are relatively self-contained to the robot.
        retargeting_cost(
            var_Ts_world_root,
            var_joints,
            var_smpl_joints_scale,
            target_keypoints,
        ),
        scale_regularization(var_smpl_joints_scale),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            100.0,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([0.2]),
        ),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full(var_joints.default_factory().shape, 0.2)
            .at[joints_to_move_less]
            .set(2.0)[None],
        ),
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], robot_coll),
            var_joints,
            margin=0.05,
            weight=2.0,
        ),
        # Costs that are scene-centric.
        pc_alignment_cost(
            var_Ts_world_root,
            var_joints,
            target_keypoints,
        ),
        floor_contact_cost(
            var_Ts_world_root,
            var_joints,
            var_offset,
            is_left_foot_contact,
            is_right_foot_contact,
            left_foot_keypoints,
            right_foot_keypoints,
        ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        skating_cost(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            OffsetVar(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            OffsetVar(jnp.arange(0, timesteps - 1)),
            is_left_foot_contact[:-1],
            is_right_foot_contact[:-1],
        ),
        world_collision_cost(
            var_Ts_world_root,
            var_joints,
            var_offset,
        ),
    ]

    solution = (
        jaxls.LeastSquaresProblem(
            costs, [var_joints, var_Ts_world_root, var_smpl_joints_scale, var_offset]
        )
        .analyze()
        .solve()
    )
    transform = solution[var_Ts_world_root]
    offset = solution[var_offset]
    transform = jaxlie.SE3.from_translation(offset) @ transform
    return transform, solution[var_joints]

def create_conn_tree(robot: pk.Robot, link_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Create a NxN connectivity matrix for N links.
    The matrix is marked Y if there is a direct kinematic chain connection
    between the two links, without bypassing the root link.
    """
    n = len(link_indices)
    conn_matrix = jnp.zeros((n, n))

    def is_direct_chain_connection(idx1: int, idx2: int) -> bool:
        """Check if two joints are connected in the kinematic chain without other retargeted joints between"""
        joint1 = link_indices[idx1]
        joint2 = link_indices[idx2]

        # Check path from joint2 up to root
        current = joint2
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint1:
                return True
            if parent in link_indices:
                # Hit another retargeted joint before finding joint1
                break
            current = parent

        # Check path from joint1 up to root
        current = joint1
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint2:
                return True
            if parent in link_indices:
                # Hit another retargeted joint before finding joint2
                break
            current = parent

        return False

    # Build symmetric connectivity matrix
    for i in range(n):
        conn_matrix = conn_matrix.at[i, i].set(1.0)  # Self-connection
        for j in range(i + 1, n):
            if is_direct_chain_connection(i, j):
                conn_matrix = conn_matrix.at[i, j].set(1.0)
                conn_matrix = conn_matrix.at[j, i].set(1.0)

    return conn_matrix


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

# When loaded from `g1_description`s 23-dof model.
G1_LINK_NAMES = [
    "pelvis",
    "pelvis_contour_link",
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
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_roll_link",
    "logo_link",
    "imu_link",
    "left_palm_link",
    "left_zero_link",
    "left_one_link",
    "left_two_link",
    "left_three_link",
    "left_four_link",
    "left_five_link",
    "left_six_link",
    "right_palm_link",
    "right_zero_link",
    "right_one_link",
    "right_two_link",
    "right_three_link",
    "right_four_link",
    "right_five_link",
    "right_six_link",
]


# Written for the 27-dof model.

G1_27_dof =[
'pelvis', 
'pelvis_contour_link',
'left_hip_pitch_link',
'left_hip_roll_link', 
'left_hip_yaw_link',
 'left_knee_link', 
 'left_ankle_pitch_link',
  'left_ankle_roll_link', 
 'right_hip_pitch_link',
  'right_hip_roll_link', 
 'right_hip_yaw_link', 
 'right_knee_link',
  'right_ankle_pitch_link',
   'right_ankle_roll_link',
   'waist_yaw_link',
    'waist_roll_link',
     'torso_link', 
   'logo_link',
    'head_link',
     'waist_support_link',
    'imu_link', 'd435_link', 'mid360_link', 
    'left_shoulder_pitch_link', 
    'left_shoulder_roll_link',
     'left_shoulder_yaw_link',
      'left_elbow_link', 'left_wrist_roll_link',
      'left_wrist_pitch_link', 'left_wrist_yaw_link',
       'left_rubber_hand', 'right_shoulder_pitch_link', 
       'right_shoulder_roll_link', 'right_shoulder_yaw_link',
        'right_elbow_link', 'right_wrist_roll_link', 
        'right_wrist_pitch_link', 
'right_wrist_yaw_link', 'right_rubber_hand'
]


def get_humanoid_retarget_indices_g1_27dof() -> tuple[jnp.ndarray, jnp.ndarray]:
    smpl_joint_retarget_indices_to_g1 = []
    g1_joint_retarget_indices = []

    for smpl_name, g1_name in [
        ("pelvis", "pelvis_contour_link"),
        ("left_hip", "left_hip_pitch_link"),
        ("right_hip", "right_hip_pitch_link"),
        ("left_knee", "left_knee_link"),
        ("right_knee", "right_knee_link"),
        ("left_ankle", "left_ankle_roll_link"),
        ("right_ankle", "right_ankle_roll_link"),
        ("left_shoulder", "left_shoulder_roll_link"),
        ("right_shoulder", "right_shoulder_roll_link"),
        ("left_elbow", "left_elbow_link"),
        ("right_elbow", "right_elbow_link"),
        ("left_wrist", "left_wrist_yaw_link"),
        ("right_wrist", "right_wrist_yaw_link"),
    ]:
        smpl_joint_retarget_indices_to_g1.append(SMPL_JOINT_NAMES.index(smpl_name))
        g1_joint_retarget_indices.append(G1_27_dof.index(g1_name))

    smpl_joint_retarget_indices = jnp.array(smpl_joint_retarget_indices_to_g1)
    g1_joint_retarget_indices = jnp.array(g1_joint_retarget_indices)
    return smpl_joint_retarget_indices, g1_joint_retarget_indices



def get_humanoid_retarget_indices() -> tuple[jnp.ndarray, jnp.ndarray]:
    smpl_joint_retarget_indices_to_g1 = []
    g1_joint_retarget_indices = []

    for smpl_name, g1_name in [
        ("pelvis", "pelvis_contour_link"),
        ("left_hip", "left_hip_pitch_link"),
        ("right_hip", "right_hip_pitch_link"),
        ("left_knee", "left_knee_link"),
        ("right_knee", "right_knee_link"),
        ("left_ankle", "left_ankle_roll_link"),
        ("right_ankle", "right_ankle_roll_link"),
        ("left_shoulder", "left_shoulder_roll_link"),
        ("right_shoulder", "right_shoulder_roll_link"),
        ("left_elbow", "left_elbow_pitch_link"),
        ("right_elbow", "right_elbow_pitch_link"),
        ("left_wrist", "left_palm_link"),
        ("right_wrist", "right_palm_link"),
    ]:
        smpl_joint_retarget_indices_to_g1.append(SMPL_JOINT_NAMES.index(smpl_name))
        g1_joint_retarget_indices.append(G1_LINK_NAMES.index(g1_name))

    smpl_joint_retarget_indices = jnp.array(smpl_joint_retarget_indices_to_g1)
    g1_joint_retarget_indices = jnp.array(g1_joint_retarget_indices)
    return smpl_joint_retarget_indices, g1_joint_retarget_indices


MANO_TO_SHADOW_MAPPING = {
    # Wrist
    0: "palm",
    # Thumb
    1: "thhub",
    2: "thmiddle",
    3: "thdistal",
    4: "thtip",
    # Index
    5: "ffproximal",
    6: "ffmiddle",
    7: "ffdistal",
    8: "fftip",
    # Middle
    9: "mfproximal",
    10: "mfmiddle",
    11: "mfdistal",
    12: "mftip",
    # Ring
    13: "rfproximal",
    14: "rfmiddle",
    15: "rfdistal",
    16: "rftip",
    # # Little
    17: "lfproximal",
    18: "lfmiddle",
    19: "lfdistal",
    20: "lftip",
}


def get_mapping_from_mano_to_shadow(robot: pk.Robot) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the mapping indices between MANO and Shadow Hand joints."""
    SHADOW_TO_MANO_MAPPING = {v: k for k, v in MANO_TO_SHADOW_MAPPING.items()}
    shadow_joint_idx = []
    mano_joint_idx = []
    link_names = robot.links.names
    for i, link_name in enumerate(link_names):
        if link_name in SHADOW_TO_MANO_MAPPING:
            shadow_joint_idx.append(i)
            mano_joint_idx.append(SHADOW_TO_MANO_MAPPING[link_name])

    return jnp.array(shadow_joint_idx), jnp.array(mano_joint_idx)
