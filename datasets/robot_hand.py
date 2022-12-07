import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import pickle

class RobotHandDataset:
    def __init__(self, split, dataroot, mode='full'):
        self.split = split
        self.dataroot = dataroot

        self.transforms_rgb = transforms.Compose([
            transforms.ToTensor(), 
        ])

        self.transforms_depth = self._normalize_depth_global

        self.field_ids = []
        self.base_path = os.path.join(self.dataroot, self.split)

        self.dataset_len = len(os.listdir(os.path.join(self.base_path, 'X')))
        self.train_size = int(self.dataset_len * 0.7)
        self.mode = mode
    
    def _normalize_depth_global(self, depth_img):
        depth_img = depth_img / 1000
        min_depth = 0
        max_depth = 65.535
        depth_img = (depth_img - min_depth) / (max_depth - min_depth)
        return depth_img

    def _normalize_depth_per_image(self, depth_img):
        depth_img = depth_img / 1000
        min_depth = np.min(depth_img)
        max_depth = np.max(depth_img)
        depth_img = (depth_img - min_depth) / (max_depth - min_depth)
        return depth_img

    def __getitem__(self, idx):
        if self.mode == 'tail':
            idx += self.train_size

        base_path_x = os.path.join(self.base_path, 'X', str(idx))
        
        image_input = None

        for ii in [0,1,2]:
            rgb_img = Image.open(os.path.join(base_path_x, f'rgb/{ii}.png'))
            rgb_img = self.transforms_rgb(rgb_img)

            depth_img = np.load(os.path.join(base_path_x, 'depth.npy'))[ii]
            depth_img = self.transforms_depth(depth_img)
            depth_img = torch.from_numpy(depth_img).unsqueeze(0)
            if image_input == None:
                image_input = torch.cat((rgb_img, depth_img), dim=0)
            else:
                image_input = torch.cat((image_input, rgb_img, depth_img), dim=0)

        # field id
        if len(self.field_ids) != self.dataset_len:
            with open(os.path.join(base_path_x, 'field_id.pkl'), 'rb') as f:
                self.field_ids.append(int(pickle.load(f)))

        # label
        base_path_y = os.path.join(self.base_path, 'Y', f'{idx}.npy')
        label = torch.Tensor([])
        if self.split == 'train':
            label = np.load(os.path.join(base_path_y))
            label = torch.from_numpy(label)

        return image_input.float(), label.float()
    
    def __len__(self):
        if self.mode == 'full':
            return self.dataset_len
        elif self.mode == 'head':
            return self.train_size
        else:
            return self.dataset_len - self.train_size