import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import pytorch_kinematics as pk
import numpy as np
import pickle as pkl
import viser
from viser.extras import ViserUrdf
import jax.numpy as jnp
import jaxlie
import yourdfpy
import pyroki as pk2
import pickle
import time
import trimesh

import torch.optim as optim
from isaac_utils.rotations import(
    quaternion_to_matrix,
    slerp
)

# Load URDF and create kinematic chain
urdf_path = '../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
chain = pk.build_chain_from_urdf(open(urdf_path).read())
pkl_path = '0.pkl'
SMOOTH_AMT = 20
PAUSE_AMT = 10
INTERP_AMT = 10
REDUCE_PAUSE_AMT = 0
SAVE_DIR = "../Ground_Pick_sim2/"
VIS = False
motion_data = pkl.load(open(pkl_path, 'rb'))

target_trans = torch.tensor(np.array(motion_data['global_position'])).clone()
target_quats = torch.tensor(np.array(motion_data['global_pose'].rotation().wxyz)).clone()

# Target joint angles (example, replace with your actual target)
target_joint_angles = torch.tensor(np.array(motion_data['joints'])).clone()
grab_pos = torch.tensor(np.array(motion_data['ground_grab_pos'])).clone()

joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
init_joint_angles = [-0.2, 0., 0., 0.42, -0.23, 0., -0.2, 0., 0., 0.42, -0.23, 0., 0., 0.35, 0.16, 0., 0.87, 0., 0., 0., 0.35, -0.16, 0., 0.87, 0., 0., 0.]
heightmap = np.zeros((1000, 1000), dtype=np.float32)  # Dummy heightmap for visualization
global_pose, joints = motion_data['global_pose'], motion_data['joints']
joints_new_ = joints.clone()  # Copy the original joints
num_timesteps = joints.shape[0]
joints_new = torch.zeros((joints_new_.shape[0]+INTERP_AMT+PAUSE_AMT, joints_new_.shape[1]), dtype=joints_new_.dtype)
trans_new = torch.zeros((joints_new_.shape[0]+INTERP_AMT+PAUSE_AMT, 3), dtype=joints_new_.dtype)
quats_new = torch.zeros((joints_new_.shape[0]+INTERP_AMT+PAUSE_AMT, 4), dtype=joints_new_.dtype)
trans_new[INTERP_AMT + PAUSE_AMT:, :] = target_trans.clone()
quats_new[INTERP_AMT + PAUSE_AMT:, :] = target_quats.clone()
joints_new[INTERP_AMT + PAUSE_AMT:, :] = joints_new_.clone()
quats_new[:INTERP_AMT + PAUSE_AMT, 0] = 1.0  # Set the first quaternion component to 1.0


joints_new[:PAUSE_AMT, :] = torch.tensor(init_joint_angles).unsqueeze(0).repeat(PAUSE_AMT, 1)
# import pdb; pdb.set_trace()
joints_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = joints_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
    (joints_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - joints_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
    torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)

heightmap = pk2.collision.Heightmap(
    pose=jaxlie.SE3.identity(),
    size=jnp.array([0.01, 0.01, 1.0]),
    height_data=heightmap,
)


# asset_dir = Path(__file__).parent / "retarget_helpers" / "humanoid" / "amass"
urdf = yourdfpy.URDF.load('../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')

server = viser.ViserServer()
base_frame = server.scene.add_frame("/base", show_axes=False)
base_frame_new = server.scene.add_frame("/base_new", show_axes=False)
urdf_vis_new = ViserUrdf(server, urdf, root_node_name="/base_new")
playing = server.gui.add_checkbox("playing", False)
timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1 + INTERP_AMT + PAUSE_AMT - REDUCE_PAUSE_AMT, 1, 0)
server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())



first_viol_i = 0
q_dict = {name: joints_new[:, i] for i, name in enumerate( joint_names ) }
rot_matrix = quaternion_to_matrix(quats_new)
fk_results = chain.forward_kinematics(q_dict)
pos_right_ankle = fk_results["right_ankle_roll_link"].get_matrix()[:,:3,3]  # shape (N, 3)
pos_right_ankle = torch.bmm(pos_right_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans_new
pos_left_ankle = fk_results["left_ankle_roll_link"].get_matrix()[:,:3,3]  # shape (N, 3)
pos_left_ankle = torch.bmm(pos_left_ankle.unsqueeze(1), rot_matrix.transpose(2, 1))[:,0] + trans_new

if pos_right_ankle[PAUSE_AMT+INTERP_AMT,0] < pos_left_ankle[PAUSE_AMT+INTERP_AMT,0]:
    trans_new[:PAUSE_AMT, 2] -= pos_right_ankle[:PAUSE_AMT, 2]
    trans_new[:PAUSE_AMT, :2] = pos_right_ankle[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :2] - pos_right_ankle[:PAUSE_AMT, :2]
    trans_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = trans_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
        (trans_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - trans_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
        torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)
else :
    trans_new[:PAUSE_AMT, 2] -= pos_left_ankle[:PAUSE_AMT, 2]
    trans_new[:PAUSE_AMT, :2] = pos_left_ankle[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :2] - pos_left_ankle[:PAUSE_AMT, :2]
    trans_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = trans_new[PAUSE_AMT-1:PAUSE_AMT, :] + \
        (trans_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :] - trans_new[PAUSE_AMT-1:PAUSE_AMT, :]) * \
        torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)

# Interpolate quats from PAUSE_AMT to PAUSE_AMT + INTERP_AMT

quats_new[PAUSE_AMT:PAUSE_AMT+INTERP_AMT, :] = slerp(
    quats_new[PAUSE_AMT-1:PAUSE_AMT, :],
    quats_new[PAUSE_AMT+INTERP_AMT:PAUSE_AMT+INTERP_AMT+1, :],
    torch.linspace(0, 1, INTERP_AMT).unsqueeze(1)
)

pick_t = PAUSE_AMT + INTERP_AMT + 5
for i in range(pick_t, trans_new.shape[0]):
    if trans_new[i, 0] == trans_new[i+1, 0] and trans_new[i, 1] == trans_new[i+1, 1] and trans_new[i, 2] == trans_new[i+1, 2]:
        pick_t = i
        break
pick_t_end = pick_t 

for i in range(pick_t_end, trans_new.shape[0]):
    if trans_new[i, 0] != trans_new[i+1, 0] or trans_new[i, 1] != trans_new[i+1, 1] or trans_new[i, 2] != trans_new[i+1, 2]:
        pick_t_end = i
        break

if REDUCE_PAUSE_AMT > 0:
    trans_new[pick_t_end-REDUCE_PAUSE_AMT:-REDUCE_PAUSE_AMT] = trans_new[pick_t_end:].clone()
    quats_new[pick_t_end-REDUCE_PAUSE_AMT:-REDUCE_PAUSE_AMT] = quats_new[pick_t_end:].clone()
    joints_new[pick_t_end-REDUCE_PAUSE_AMT:-REDUCE_PAUSE_AMT] = joints_new[pick_t_end:].clone()
    trans_new = trans_new[:-REDUCE_PAUSE_AMT]
    quats_new = quats_new[:-REDUCE_PAUSE_AMT]
    joints_new = joints_new[:-REDUCE_PAUSE_AMT]

Ts_world_root = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(jnp.array(quats_new)),jnp.array(trans_new))

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

with open(SAVE_DIR + pkl_path[:-4] + "_n.pkl","wb") as f:
    pickle.dump({"global_pose": Ts_world_root , "joints": joints_new, "global_position": trans_new, "ground_grab_pos": motion_data["ground_grab_pos"], "grab_idx": motion_data["grab_idx"]+PAUSE_AMT+INTERP_AMT}, f)

if VIS :
    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
                time.sleep(0.1)
            tstep = timestep_slider.value
            base_frame_new.wxyz = np.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame_new.position = np.array(Ts_world_root.wxyz_xyz[tstep][4:]) + np.array([0, 0, 0.035])  # Adjust for the height of the robot's base
            urdf_vis_new.update_cfg(np.array(joints_new[tstep]))
