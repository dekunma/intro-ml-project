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

        self.transforms_rgb1 = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transforms_rgb2 = transforms.Compose([
            transforms.Normalize(
                mean = [0.396, 0.417, 0.435, 0.470, 0.488, 0.501, 0.441, 0.482, 0.519],
                std = [0.212, 0.199, 0.200, 0.242, 0.225, 0.228, 0.247, 0.229, 0.230])
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

        # normalize to [-1, 1]
        # depth_img = depth_img * 2 - 1

        # if self.split == 'train':
        #     rand_dot_percentage = 0.15
        #     p = np.random.rand() * rand_dot_percentage
        #     depth_img = self._pix_dropout(depth_img, background_value=1, P=p, V=1)

        return depth_img

    def _normalize_depth_per_image(self, depth_img):
        depth_img = depth_img / 1000
        min_depth = np.min(depth_img)
        max_depth = np.max(depth_img)
        depth_img = (depth_img - min_depth) / (max_depth - min_depth)

        return depth_img

    def _pix_dropout(self, img,background_value,P,V=1):
        # image is a tensor of size (1,H,W)
        # background_value denotes the value using which the foreground pixels are computed
        # P is the percentage of foreground pixels that are assigned the value V

        # returns the same image with P percentage of its foregrounds pixels set to the value V

        img = img.clone()
        mask=abs(img[0]-background_value)>1e-6 # locate foreground pixels
        y,x=np.where(mask) # get their pixel coordinates

        num_pix=np.int32(P*len(y))
        indecies_toSet=np.random.choice(len(x),size=num_pix,replace=False)
        img[0,y[indecies_toSet],x[indecies_toSet]]=V

        return img

    def __getitem__(self, idx):
        if self.mode == 'tail':
            idx += self.train_size

        base_path_x = os.path.join(self.base_path, 'X', str(idx))
        
        rgb_imgs = [Image.open(os.path.join(base_path_x, f'rgb/{ii}.png')) for ii in range(3)]
        rgb_imgs = [self.transforms_rgb1(img) for img in rgb_imgs]
        rgb_imgs = torch.cat(rgb_imgs, dim=0)
        rgb_imgs = self.transforms_rgb2(rgb_imgs)

        depth_img = np.load(os.path.join(base_path_x, 'depth.npy'))
        depth_img = torch.from_numpy(depth_img)
        depth_img = self.transforms_depth(depth_img)
        image_input = torch.cat((rgb_imgs, depth_img), dim=0)

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