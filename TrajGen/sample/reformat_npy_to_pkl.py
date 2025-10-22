import numpy as np

data = np.load('/home/azureuser/IsaacLab/TrajGen/save/omnicontrol_ckpt/samples_omnicontrol_ckpt__humanml3d_seed10_predefined/results.npy', allow_pickle=True).item()

print(data['motion'].shape)