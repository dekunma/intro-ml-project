import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import pickle

class RobotHandDepthDataset:
    def __init__(self, split, dataroot, mode='full'):
        self.split = split
        self.dataroot = dataroot

        self.transforms = transforms.Compose([
            transforms.ToTensor(), 
        ])

        self.field_ids = []
        self.base_path = os.path.join(self.dataroot, self.split)

        self.dataset_len = len(os.listdir(os.path.join(self.base_path, 'X')))
        self.train_size = int(self.dataset_len * 0.7)
        self.mode = mode
    
    def __getitem__(self, idx):
        if self.mode == 'tail':
            idx += self.train_size

        base_path_x = os.path.join(self.base_path, 'X', str(idx))
        
        image_input = None

        # TODO: try using the other two angles
        for ii in [0]:
            depth_img = np.load(os.path.join(base_path_x, 'depth.npy'))[ii]
            depth_img = Image.fromarray(depth_img)
            depth_img = self.transforms(depth_img)
            if image_input == None:
                image_input = depth_img
            else:
                image_input = torch.cat((image_input, depth_img), dim=0)

        # field id
        if len(self.field_ids) != self.dataset_len:
            with open(os.path.join(base_path_x, 'field_id.pkl'), 'rb') as f:
                self.field_ids.append(int(pickle.load(f)))

        # label
        base_path_y = os.path.join(self.base_path, 'Y', '0.npy')
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