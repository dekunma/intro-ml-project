import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

transforms_rgb = transforms.Compose([
    transforms.ToTensor(), 
                transforms.Normalize(
                mean = [0.441, 0.482, 0.519],
                std = [0.247, 0.229, 0.230])
])

base_dir = '/scratch/dm4524/data/robot_hand/train/X'

for ii in range(2,3):
    for channel in range(3):
        all_x = []
        for file in tqdm(os.listdir(base_dir)):
            current_x_dir = os.path.join(base_dir, file)
            rgb_img = Image.open(os.path.join(current_x_dir, f'rgb/{ii}.png'))
            rgb_img = transforms_rgb(rgb_img)
            rgb_img = rgb_img.numpy()
            all_x.append(rgb_img[channel])
    
        print(f'File: {ii}, Channel: {channel}, mean: {np.mean(all_x)}, std: {np.std(all_x)}')

    #     current_x = np.load(os.path.join(current_x_dir, 'depth.npy'))[channel] / 1000
    #     current_min = np.min(current_x)
    #     current_max = np.max(current_x)
    #     # all_x.append(current_x)
    #     all_min.add(current_min)
    #     all_max.add(current_max)

    # # all_x = np.array(all_x) / 1000
    # # print(f'Channel {ii}, min: {np.min(all_x)}, max: {np.max(all_x)}')
    # print(f'Channel {channel}, mins: {all_min}, max: {all_max}')
    # print(f'Channel {channel}, global max: {max(all_max)}')
