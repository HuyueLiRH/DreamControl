import time

import jax.numpy as jnp
import jaxlie
import numpy as onp
import pyroki as pk
import viser
from viser.extras import ViserUrdf
import pickle
import yourdfpy
import pyroki as pk

import argparse

def load_pickle(path):
    with open(path, "rb") as f:
        DATA = pickle.load(f)
    
    return DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="Pick_sim2")
    parser.add_argument("--id", type=str, default="0")
    args = parser.parse_args()
    g1_data_path = f"{args.folder}/{args.id}.pkl"
    urdf = yourdfpy.URDF.load('../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')

    G1_DATA = load_pickle(g1_data_path)
    heightmap = onp.zeros((1000, 1000), dtype=onp.float32)  # Dummy heightmap for visualization
    global_pose, joints = G1_DATA['global_pose'], G1_DATA['joints']
    num_timesteps = joints.shape[0]

    heightmap = pk.collision.Heightmap(
        pose=jaxlie.SE3.identity(),
        size=jnp.array([0.01, 0.01, 1.0]),
        height_data=heightmap,
    )

    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True)
    timestep_slider = server.gui.add_slider("timestep", 0, num_timesteps - 1, 1, 0)
    server.scene.add_mesh_trimesh("/heightmap", heightmap.to_trimesh())

    Ts_world_root, joints = global_pose , joints 
    positions = onp.array(Ts_world_root.translation())
    orientations = onp.array(Ts_world_root.rotation().wxyz)


    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            base_frame.wxyz = orientations[tstep]
            base_frame.position = positions[tstep] + onp.array([0, 0, 0.035])  # Adjust for the height of the robot's base
            urdf_vis.update_cfg(onp.array(joints[tstep]))

        time.sleep(1./20.)



if __name__ == "__main__":
    main()


