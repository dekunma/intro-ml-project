import numpy as np
import matplotlib.pyplot as plt
import torch

depth = np.load('/scratch/dm4524/data/robot_hand/train/X/0/depth.npy')


def _normalize_depth_per_image(depth_img):
    depth_img = depth_img / 1000
    min_depth = np.min(depth_img)
    max_depth = np.max(depth_img)
    depth_img = (depth_img - min_depth) / (max_depth - min_depth)

    return depth_img

def _pix_dropout(img,background_value,P,V=1):
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

rand_dot_percentage = 0.15
p = np.random.rand() * rand_dot_percentage

depth_img = _normalize_depth_per_image(depth)

plt.imshow(depth_img[0])
plt.show()
plt.savefig('original.png')

depth_img = torch.from_numpy(depth_img)
depth_img = _pix_dropout(depth_img, background_value=1, P=p, V=1)
plt.imshow(depth_img[0])
plt.show()
plt.savefig('dropped.png')