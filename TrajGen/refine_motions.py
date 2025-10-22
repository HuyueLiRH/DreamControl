import numpy as np
import os

def refine_motions(motion_name):
    motion = np.load(f"motions/{motion_name}.npy")
    motion = motion.reshape(1000, -1, 3)
    motion = motion[::2]
    np.save(f"motions/{motion_name}.npy", motion)

if __name__ == "__main__":
    refine_motions("Pick-a")