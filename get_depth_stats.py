import os
import numpy as np
from tqdm import tqdm


for ii in range(3):
    base_dir = '/scratch/dm4524/data/robot_hand/train/X'
    all_x = []
    for file in tqdm(os.listdir(base_dir)):
        current_x_dir = os.path.join(base_dir, file)
        current_x = np.load(os.path.join(current_x_dir, 'depth.npy'))[ii]
        all_x.append(current_x)

    all_x = np.array(all_x) / 1000
    print(f'Channel {ii}, min: {np.min(all_x)}, max: {np.max(all_x)}')
