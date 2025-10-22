import time
from typing import Tuple, TypedDict
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
# import jaxls
import numpy as onp
import pyroki as pk
import pytorch_kinematics as pk2
import trimesh
import viser
from viser.extras import ViserUrdf
import glob
import pickle
import yourdfpy
import torch
from isaac_utils.rotations import(
    quat_conjugate,
    quaternion_to_matrix
)
import pyroki as pk

VIS_ROBOT = True
VIS_HUMAN = True
VIS_TABLE = False

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

def load_pickle(path):
    
    with open(path, "rb") as f:
        DATA = pickle.load(f)
    
    return DATA

def get_keypts(joint_angles, joint_names, pk2_robot):
    
        q_dict = {name: joint_angles[:, i] for i, name in enumerate( joint_names ) }

        tf_dict = pk2_robot.forward_kinematics( q_dict )

        keypts = torch.zeros((len(joint_angles), len(tf_dict), 3))
        # print(len(tf_dict))
        # exit(0)
        cntr = 0 
        
        for name in tf_dict.keys():
            # print("1, #", name)
            tf_val = tf_dict[name].get_matrix()  
            t = tf_val[:, :3, -1 ]
            # print(name, t)
            keypts[:, cntr , :] = t 
            cntr += 1 
        # print(keypts)
        # exit(0)
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

def main():

    urdf = yourdfpy.URDF.load('../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    robot = pk.Robot.from_urdf(urdf)
    
    
    print("1. Loading data...")
    heightmap = onp.zeros((1000, 1000), dtype=onp.float32)  # Dummy heightmap for visualization
    num_timesteps = 195

    heightmap = pk.collision.Heightmap(
        pose=jaxlie.SE3.identity(),
        size=jnp.array([0.01, 0.02, 1.0]),
        height_data=heightmap,
    )


    # asset_dir = Path(__file__).parent / "retarget_helpers" / "humanoid" / "amass"

    server = viser.ViserServer(port=8083)
    
    playing = server.gui.add_checkbox("playing", False)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())
    g1_data_paths = glob.glob("pick_final2/*.pkl")
    positionss = []
    orientationss = []
    jointss = []
    urdf_vis_list = []
    base_frames = []
    idx = 0
    offsets = []
    pick_ts = []
    for i in range(12):
        x = int(i/3)*2. - 4.
        y = int(i%3)*2. - 3.
        offsets.append(onp.array([x, y, 0.035]))
    
    DATA = onp.load('pick_final/results.npy', allow_pickle=True).item()
    smpl_keypoints_all = DATA['motion'].transpose((0,3,1,2)) # Shape: (k, N, 22, 3)
    rot_mat = onp.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]])
    smpl_keypoints = onp.einsum('ij, ankj -> anki', rot_mat, smpl_keypoints_all.copy()) # Shape: (k, N, 22, 3)

    for i in range(12):
        motion_n = int(g1_data_paths[i].split("/")[-1][:-6])
        smpl_keypoints[motion_n] += offsets[i][None, None, :]
    smpl_keypoints = smpl_keypoints[:,1:]
    
    for g1_data_path in g1_data_paths[:12]:
        idx += 1
        g1_data_path_old = g1_data_path.replace("pick_final2", "pick_final1").replace("_n.", ".")
        # G1_DATA = load_pickle(g1_data_path)
        G1_DATA = load_pickle(g1_data_path_old)
        motion_n = int(g1_data_path.split("/")[-1][:-6])
        pick_ts.append(G1_DATA['grab_idx'])
        if VIS_ROBOT:
            base_frame = server.scene.add_frame("/base"+str(idx), show_axes=False)
            base_frames.append(base_frame)

        global_pose, joints = G1_DATA['global_pose'], G1_DATA['joints']
        if VIS_ROBOT:
            urdf_vis = ViserUrdf(server, urdf, root_node_name="/base"+str(idx))
            urdf_vis_list.append(urdf_vis)
        Ts_world_root, joints = global_pose , joints 
        positions = onp.array(Ts_world_root.translation())
        orientations = onp.array(Ts_world_root.rotation().wxyz)

        joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        urdf_path = '../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
        pk2_robot = pk2.build_chain_from_urdf(open(urdf_path).read())
        
        keypts = get_keypts(torch.tensor(joints), joint_names , pk2_robot=pk2_robot)


        global_keypts = transform_keypts(torch.tensor(keypts), torch.tensor(orientations), torch.tensor(positions)).numpy()

        # Define cuboid dimensions (width, height, depth)
        width, height, depth = 0.05, 0.05, 0.05

        color = onp.array([0.8, 0.0, 0.0, 1.0])  # floats in [0,1]

        # Create a cuboid using Trimesh
        cuboid = trimesh.creation.box(extents=(width, depth, height))
        
        cuboid.visual.face_colors = (color * 255).astype(onp.uint8)
        # gen_button = server.gui.add_button("Retarget!")
        transform = onp.eye(4)
        if VIS_HUMAN:
            transform[:3, 3] = [smpl_keypoints[motion_n,G1_DATA["grab_idx"]-20, -1, 0] , smpl_keypoints[motion_n,G1_DATA["grab_idx"]-20, -1, 1], smpl_keypoints[motion_n,G1_DATA["grab_idx"]-20, -1, 2]]
        if VIS_ROBOT:
            transform[:3, 3] = [global_keypts[G1_DATA["grab_idx"], -1, 0] +offsets[idx-1][0] , global_keypts[G1_DATA["grab_idx"], -1, 1]+offsets[idx-1][1], global_keypts[G1_DATA["grab_idx"], -1, 2]]
        print(motion_n,transform[:3, 3])
        cuboid.apply_transform(transform)
        
        server.scene.add_mesh_trimesh(
            name="target"+str(idx),
            mesh=cuboid,
            # wireframe=False,  # Set to True if you want to see just the wireframe
        )

        color = onp.array([0.0, 0.67, 0.5, 1.0])  # floats in [0,1]
        width, height, depth = 0.5, 0.86, 1.0
        # Create a cuboid using Trimesh
        platform = trimesh.creation.box(extents=(width, depth, height))
        
        platform.visual.face_colors = (color * 255).astype(onp.uint8)
        # gen_button = server.gui.add_button("Retarget!")
        transform = onp.eye(4)
        transform[:3, 3] = [global_keypts[G1_DATA["grab_idx"], -1, 0]+0.2 +offsets[idx-1][0] , 0.+offsets[idx-1][1], 0.43]
        platform.apply_transform(transform)
        
        # print(transform[:3, 3])
        # Add the cuboid to Viser
        if VIS_TABLE:
            server.scene.add_mesh_trimesh(
                name="platform"+str(idx),
                mesh=platform,
                # wireframe=False,  # Set to True if you want to see just the wireframe
            )
        jointss.append(joints)
        positionss.append(positions)
        orientationss.append(orientations)

    
    # print(positions)
    print("Started?")
    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            for i in range(len(jointss)):
                motion_n = int(g1_data_paths[i].split("/")[-1][:-6])
                # print(motion_n)
                joints = jointss[i][tstep]
                positions = positionss[i][tstep] + offsets[i]
                orientations = orientationss[i][tstep]
                if VIS_ROBOT:   
                    base_frame = base_frames[i]
                    base_frame.wxyz = orientations
                    base_frame.position = positions + onp.array([0, 0, 0.035])  # Adjust for the height of the robot's base
                    urdf_vis_list[i].update_cfg(onp.array(joints))

                if VIS_HUMAN:
                    skeleton = [
                        [-1, 0], [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7],
                        [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
                        [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]
                    ]
                    # --- End of dummy data ---

                    if tstep > pick_ts[i]:
                        tstep = max(tstep-10,pick_ts[i])
                    # Let's assume 'tstep' is defined elsewhere in your loop
                    current_smpl_keypoints = smpl_keypoints[motion_n,max(tstep-0,0)] # Replace with smpl_keypoints[tstep]

                    # Prepare points for the line segments in (N, 2, 3) format
                    line_segment_points_list = []
                    for bone in skeleton:
                        idx0, idx1 = bone
                        # Skip if the first index is -1
                        if idx0 == -1:
                            continue
                        # Ensure indices are within bounds
                        if 0 <= idx0 < len(current_smpl_keypoints) and 0 <= idx1 < len(current_smpl_keypoints):
                            start_point = current_smpl_keypoints[idx0]
                            end_point = current_smpl_keypoints[idx1]
                            line_segment_points_list.append([start_point, end_point])
                        else:
                            print(f"Warning: Bone {bone} has out-of-bounds indices for current_smpl_keypoints with shape {current_smpl_keypoints.shape}")

                    if line_segment_points_list:
                        # Convert to NumPy array of shape (N, 2, 3)
                        points_for_lines = onp.array(line_segment_points_list)
                        num_lines = len(points_for_lines)

                        # Prepare colors for the line segments (N, 2, 3)
                        # Let's make each segment uniformly blue. So both endpoints of a segment are blue.
                        # RGB color for blue
                        blue_color = onp.array([0, 0, 255])
                        # Create an array (N, 2, 3) where each [start_color, end_color] pair is [blue, blue]
                        colors_for_lines = onp.tile(blue_color, (num_lines, 2, 1))


                        server.add_line_segments(
                            name="/target_skeleton_segments"+str(i), # Unique name
                            points=points_for_lines,
                            colors=colors_for_lines, # (N, 2, 3) array, colors for each endpoint
                            line_width=3.0, # As per your example
                        )


                    server.scene.add_point_cloud(
                        "/target_keypoints"+str(i),
                        onp.array(smpl_keypoints[motion_n,max(tstep-0,0)]),
                        onp.array((0, 0, 255))[None].repeat(22, axis=0),
                        point_size=0.01,
                    )

        time.sleep(1./2.)




if __name__ == "__main__":

    data_file_id = "005783"

    # human_data_path = "/home/gr/sudarshan/Sparky/pyroki/examples/amass_data/" + data_file_id + ".pkl"
    # g1_data_path = "000332.pkl"
    g1_data_path = "pick_final2/2_n.pkl"
    # g1_data_path = "follow1/0.pkl"
    print("Right script?")
    main()


