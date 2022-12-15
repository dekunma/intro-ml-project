import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

from utils.model_utils import name2model
from utils.dataset_utils import get_dataset

model_name = 'depth_net'
dataroot = '/scratch/dm4524/data/robot_hand'
mode = 'full'

dataset = get_dataset(model_name, 'train', dataroot, mode)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

all_depth = [[], [], []]

for imgs, labels in tqdm(data_loader):
    imgs = imgs.squeeze(0)
    for ii in range(3):
        all_depth[ii].extend(imgs[ii].flatten().numpy())

for ii in range(3):
    print(np.mean(all_depth[ii]), np.std(all_depth[ii]), np.min(all_depth[ii]), np.max(all_depth[ii]))


# for ii in range(3):
#     plt.hist(all_depth[ii], bins=100)
#     plt.title(f'Channel {ii}')
#     plt.show()
#     plt.savefig(f'channel_{ii}.png')